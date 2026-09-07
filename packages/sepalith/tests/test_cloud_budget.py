"""Budget admission and provider recovery through their public interfaces."""
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

from sepalith.cloud_budget import BudgetLedger, AdmissionError


def policy():
    return dict(currency='USD', allowance=100_000_000, production_reserve=85_000_000,
                safety_reserve=5_000_000, observed_at=1000, valid_until=1900,
                credit_expires=10000, expiry_headroom=300,
                evidence='fake verified credit; micro-USD; tax included',
                covered_components=['provisioning', 'compute', 'evaluation', 'upload',
                                    'shutdown', 'disks', 'ips', 'storage', 'egress', 'tax'])


def quote():
    return dict(currency='USD', observed_at=1000, valid_until=1900,
                evidence='fake binding total-price ceiling',
                price_ceiling_until=2000, resource_spec={'shape': 'fake-fixed-one'},
                components={name: dict(fixed=100_000, per_second=1000)
                            for name in ('provisioning', 'compute', 'evaluation', 'upload',
                                         'shutdown', 'disks', 'ips', 'storage', 'egress', 'tax')})


def request():
    return dict(runtime_seconds=100, deadline=1500, max_cost=2_000_000,
                recipe_sha256='a' * 64, resource_spec={'shape': 'fake-fixed-one'},
                retries=0, max_workers=1, restart=False)


class FakeProvider:
    """External provider simulator, independent of the durable ledger."""
    name = 'fake'

    def __init__(self):
        self.resources = {}
        self.guard_works = True
        self.launch_error = False
        self.termination_error = False
        self.final_cost = None
        self.residual = False
        self.calls = []
        self.guards = {}

    def inventory(self):
        return {'complete': True, 'resource_ids': [k for k, v in self.resources.items() if v == 'running']}

    def arm(self, job):
        guard = dict(provider_id='resource-' + job['id'], deadline=job['deadline'],
                    mechanism='fake provider independent lease', verified=self.guard_works,
                    provisioning_bounded=True, descendants_bounded=True,
                    resource_spec=job['request']['resource_spec'])
        self.guards[job['id']] = guard
        return guard

    def lookup(self, job):
        return dict(job_id=job['id'], recipe_sha256=job['request']['recipe_sha256'],
                    complete=True, evidence='fake exact submission-key lookup', **self.guards[job['id']])

    def submit(self, job):
        if not job['termination']['verified']:
            raise AssertionError('Compute before verified termination')
        self.calls.append('submit')
        self.resources[job['provider_id']] = 'running'
        if self.launch_error:
            raise TimeoutError('Launch response lost')

    def inspect(self, job):
        state = self.resources.get(job['provider_id'], 'absent')
        return dict(provider_id=job['provider_id'], terminal=state in ('terminated', 'absent'),
                    residual_billable=self.residual, final_cost=self.final_cost,
                    currency='USD', evidence='fake provider audit',
                    operational_result=state)

    def terminate(self, job):
        self.calls.append('terminate')
        if self.termination_error:
            raise TimeoutError('Provider termination unavailable')
        self.resources[job['provider_id']] = 'terminated'


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'budget.sqlite3'
        self.now = 1000
        self.ledger = BudgetLedger(self.path, clock=lambda: self.now)
        self.ledger.configure('fake', policy())
        self.ledger.reserve('job', 'fake', request(), quote())
        self.provider = FakeProvider()

    def test_guard_precedes_compute_and_final_billing_precedes_release(self):
        self.ledger.launch('job', self.provider, quote())
        self.assertEqual('running', self.ledger.get('job')['state'])
        self.now = 1101
        self.ledger.reconcile('job', self.provider)
        self.assertEqual('awaiting_billing', self.ledger.get('job')['state'])
        self.assertEqual(2_000_000, self.ledger.get('job')['reserved'])
        self.provider.final_cost = 1_500_000
        self.ledger.reconcile('job', self.provider)
        self.assertEqual('settled', self.ledger.get('job')['state'])
        self.assertEqual(1_500_000, self.ledger.get('job')['charged'])
        self.assertIsNone(self.ledger.get('job')['scientific_verdict'])

    def test_uncertain_launch_survives_restart_and_blocks_replacement(self):
        self.provider.launch_error = True
        with self.assertRaises(TimeoutError):
            self.ledger.launch('job', self.provider, quote())
        restarted = BudgetLedger(self.path, clock=lambda: self.now)
        self.assertEqual('unknown', restarted.get('job')['state'])
        restarted.reserve('replacement', 'fake', request(), quote())
        with self.assertRaisesRegex(AdmissionError, 'Uncertain'):
            restarted.launch('replacement', self.provider, quote())
        self.provider.final_cost = 100
        restarted.reconcile('job', self.provider, cancel=True)
        self.assertEqual('settled', restarted.get('job')['state'])

    def test_unverified_guard_never_creates_compute(self):
        self.provider.guard_works = False
        with self.assertRaises(AdmissionError):
            self.ledger.launch('job', self.provider, quote())
        self.assertEqual([], self.provider.calls)
        self.assertEqual('unknown', self.ledger.get('job')['state'])

    def test_external_resource_and_changed_price_block_launch(self):
        self.provider.resources['manual-worker'] = 'running'
        with self.assertRaisesRegex(AdmissionError, 'External'):
            self.ledger.launch('job', self.provider, quote())
        self.provider.resources.clear()
        q = quote()
        q['components']['compute']['per_second'] += 1
        with self.assertRaisesRegex(AdmissionError, 'Price'):
            self.ledger.launch('job', self.provider, q)
        self.assertEqual([], self.provider.calls)

    def test_failed_termination_and_residual_billing_retain_commitment(self):
        self.ledger.launch('job', self.provider, quote())
        self.provider.termination_error = True
        self.now = 1101
        with self.assertRaises(TimeoutError):
            self.ledger.reconcile('job', self.provider)
        self.assertEqual('termination_failed', self.ledger.get('job')['state'])
        self.provider.termination_error = False
        self.provider.residual = True
        self.provider.final_cost = 0
        with self.assertRaisesRegex(AdmissionError, 'residual'):
            self.ledger.reconcile('job', self.provider)
        self.assertNotIn('charged', self.ledger.get('job'))
        self.provider.residual = False
        self.ledger.reconcile('job', self.provider)
        self.assertEqual('settled', self.ledger.get('job')['state'])

    def test_stalled_setup_expires_without_submission(self):
        arm = self.provider.arm
        def slow_arm(job):
            self.now = 1200
            return arm(job)
        self.provider.arm = slow_arm
        with self.assertRaisesRegex(AdmissionError, 'Deadline'):
            self.ledger.launch('job', self.provider, quote())
        self.assertEqual([], self.provider.calls)

    def test_quota_denial_requires_provider_audit_before_release(self):
        def denied(job):
            raise RuntimeError('quota denied')
        self.provider.submit = denied
        with self.assertRaises(RuntimeError):
            self.ledger.launch('job', self.provider, quote())
        self.provider.final_cost = 0
        self.ledger.reconcile('job', self.provider)
        self.assertEqual('settled', self.ledger.get('job')['state'])

    def test_failed_cancellation_remains_requested_across_restart(self):
        self.ledger.launch('job', self.provider, quote())
        self.provider.termination_error = True
        with self.assertRaises(TimeoutError):
            self.ledger.reconcile('job', self.provider, cancel=True)
        self.provider.termination_error = False
        self.provider.final_cost = 0
        restarted = BudgetLedger(self.path, clock=lambda: 1001)
        restarted.reconcile('job', self.provider)
        self.assertEqual('settled', restarted.get('job')['state'])

    def test_incomplete_inventory_cannot_mean_empty_account(self):
        self.provider.inventory = lambda: {'complete': True}
        with self.assertRaises(AdmissionError):
            self.ledger.launch('job', self.provider, quote())

    def test_final_cost_overrun_blocks_follow_on_compute(self):
        self.ledger.launch('job', self.provider, quote())
        self.provider.final_cost = 3_000_000
        with self.assertRaisesRegex(AdmissionError, 'exceeded'):
            self.ledger.reconcile('job', self.provider, cancel=True)
        self.ledger.reserve('follow-on', 'fake', request(), quote())
        with self.assertRaisesRegex(AdmissionError, 'Uncertain'):
            self.ledger.launch('follow-on', self.provider, quote())

    def test_lost_guard_response_requires_exact_provider_identity_lookup(self):
        arm = self.provider.arm
        def lost(job):
            arm(job)
            raise TimeoutError('Guard response lost')
        self.provider.arm = lost
        with self.assertRaises(TimeoutError):
            self.ledger.launch('job', self.provider, quote())
        restarted = BudgetLedger(self.path, clock=lambda: 1001)
        restarted.recover_identity('job', self.provider)
        self.assertEqual('resource-job', restarted.get('job')['provider_id'])
        self.provider.final_cost = 0
        restarted.reconcile('job', self.provider, cancel=True)
        self.assertEqual('settled', restarted.get('job')['state'])
        self.assertNotIn('submit', self.provider.calls)

    def test_overrun_remains_blocked_when_later_billing_is_incomplete(self):
        self.ledger.launch('job', self.provider, quote())
        self.provider.final_cost = 3_000_000
        with self.assertRaises(AdmissionError):
            self.ledger.reconcile('job', self.provider, cancel=True)
        self.provider.final_cost = None
        self.ledger.reconcile('job', self.provider)
        self.assertEqual('overrun', self.ledger.get('job')['state'])

    def test_refreshed_headroom_invalidates_old_queued_deadline(self):
        p = policy()
        p['expiry_headroom'] = 8950
        self.ledger.refresh('fake', p)
        with self.assertRaises(AdmissionError):
            self.ledger.launch('job', self.provider, quote())

    def test_expired_evidence_after_arming_cannot_start_compute(self):
        # Short credit evidence lifetime, but a longer allocation deadline.
        p = policy()
        p['valid_until'] = 1001
        self.ledger.refresh('fake', p)
        arm = self.provider.arm
        def slow(job):
            result = arm(job)
            self.now = 1002
            return result
        self.provider.arm = slow
        with self.assertRaises(AdmissionError):
            self.ledger.launch('job', self.provider, quote())
        self.assertEqual([], self.provider.calls)

    def test_inspection_outage_does_not_prevent_recorded_identity_cleanup(self):
        self.ledger.launch('job', self.provider, quote())
        def unavailable(job):
            raise TimeoutError('Inventory service unavailable')
        self.provider.inspect = unavailable
        with self.assertRaises(TimeoutError):
            self.ledger.reconcile('job', self.provider, cancel=True)
        self.assertIn('terminate', self.provider.calls)
        self.assertNotIn('charged', self.ledger.get('job'))

    def test_never_launched_commitment_can_be_cancelled_atomically(self):
        self.ledger.cancel_reserved('job')
        self.assertEqual('settled', self.ledger.get('job')['state'])
        self.assertEqual(0, self.ledger.get('job')['charged'])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'budget.sqlite3'
        self.ledger = BudgetLedger(self.path, clock=lambda: 1000)
        self.ledger.configure('fake', policy())

    def test_commitments_preserve_production_and_safety_reserves_after_restart(self):
        for i in range(5):
            self.ledger.reserve(str(i), 'fake', request(), quote())
        restarted = BudgetLedger(self.path, clock=lambda: 1000)
        with self.assertRaisesRegex(AdmissionError, 'allowance'):
            restarted.reserve('sixth', 'fake', request(), quote())
        row = restarted.get('0')
        self.assertEqual(2_000_000, row['reserved'])
        self.assertEqual(1100, row['deadline'])
        self.assertEqual('reserved', row['state'])

    def test_unknown_or_stale_cost_bounds_fail_closed(self):
        cases = [
            ('currency', 'EUR'), ('valid_until', 999), ('observed_at', 1001),
            ('evidence', ''), ('components', {}),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                q = quote()
                q[field] = value
                with self.assertRaises(AdmissionError):
                    self.ledger.reserve(field, 'fake', request(), q)
        for value in (True, 0.01, -1, '1000'):
            with self.subTest(value=value):
                q = quote()
                q['components']['compute']['per_second'] = value
                with self.assertRaises(AdmissionError):
                    self.ledger.reserve('units', 'fake', request(), q)

    def test_concurrent_admission_cannot_spend_the_same_allowance(self):
        def reserve(i):
            ledger = BudgetLedger(self.path, clock=lambda: 1000)
            try:
                ledger.reserve(str(i), 'fake', request(), quote())
                return True
            except AdmissionError:
                return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(5, sum(pool.map(reserve, range(12))))

    def test_runtime_credit_and_retry_bounds_fail_closed(self):
        for field, value in [('runtime_seconds', None), ('deadline', 1001),
                             ('max_cost', 1), ('retries', 1), ('restart', True), ('max_workers', 2)]:
            with self.subTest(field=field):
                r = request()
                r[field] = value
                with self.assertRaises(AdmissionError):
                    self.ledger.reserve(field, 'fake', r, quote())
        self.ledger.clock = lambda: 1901
        with self.assertRaisesRegex(AdmissionError, 'Stale'):
            self.ledger.reserve('stale', 'fake', request(), quote())

    def test_prices_must_cover_selected_resource_and_entire_lifetime(self):
        for key, value in [('resource_spec', {'shape': 'unpriced-expensive'}),
                           ('price_ceiling_until', 1050)]:
            q = quote()
            q[key] = value
            with self.subTest(key=key), self.assertRaises(AdmissionError):
                self.ledger.reserve(key, 'fake', request(), q)
        p = policy()
        p['covered_components'].remove('egress')
        self.ledger.configure('fake', p)
        with self.assertRaises(AdmissionError):
            self.ledger.reserve('uncovered', 'fake', request(), quote())

    def test_refresh_cannot_replenish_credit_or_reduce_production_reserve(self):
        self.ledger.reserve('existing', 'fake', request(), quote())
        refreshed = policy()
        refreshed.update(observed_at=1100, valid_until=2000, allowance=200_000_000,
                         production_reserve=0, safety_reserve=1)
        self.ledger.clock = lambda: 1100
        self.ledger.refresh('fake', refreshed)
        q = quote()
        q.update(observed_at=1100, valid_until=2000)
        for i in range(4):
            self.ledger.reserve('after' + str(i), 'fake', request(), q)
        with self.assertRaisesRegex(AdmissionError, 'allowance'):
            self.ledger.reserve('too-many', 'fake', request(), q)


if __name__ == '__main__':
    unittest.main()

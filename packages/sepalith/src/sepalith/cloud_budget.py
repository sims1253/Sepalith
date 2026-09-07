"""Durable cloud admission. Amounts are integer millionths of one named currency.

No real provider is enabled by this module. Provider adapters must establish
account-wide inventory and independently enforced allocation deadlines before
submission. A ledger is an operational bound, not a financial hard stop.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import re
import fcntl

COMPONENTS = {'provisioning', 'compute', 'evaluation', 'upload', 'shutdown',
              'disks', 'ips', 'storage', 'egress', 'tax'}


class AdmissionError(ValueError):
    """Admission is closed or an existing allocation needs investigation."""


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise AdmissionError(f'{name} must be an integer >= {minimum}')
    return value


def evidence(record, now):
    if not isinstance(record.get('evidence'), str) or not record['evidence'].strip():
        raise AdmissionError('Missing price/credit evidence')
    start = integer(record.get('observed_at'), 'observed_at')
    end = integer(record.get('valid_until'), 'valid_until')
    if not start <= now < end or end - start > 900:
        raise AdmissionError('Stale or future price/credit evidence; maximum age is 900 seconds')
    if record.get('currency') not in ('USD', 'EUR'):
        raise AdmissionError('Explicit USD or EUR currency required; no conversion')


def validate_bounds(policy, request, quote, now):
    evidence(policy, now)
    evidence(quote, now)
    if policy['currency'] != quote['currency']:
        raise AdmissionError('Currency mismatch')
    if (not isinstance(policy.get('covered_components'), list)
            or set(policy['covered_components']) != COMPONENTS):
        raise AdmissionError('Credit coverage is not verified for every lifecycle cost')
    for key in ('allowance', 'production_reserve', 'safety_reserve', 'credit_expires', 'expiry_headroom'):
        integer(policy.get(key), key, 1 if key in ('safety_reserve', 'expiry_headroom') else 0)
    if policy['production_reserve'] + policy['safety_reserve'] > policy['allowance']:
        raise AdmissionError('Reserves exceed allowance')
    for key in ('runtime_seconds', 'deadline', 'max_cost'):
        integer(request.get(key), key, 1)
    if (type(request.get('retries')) is not int or request['retries'] != 0
            or type(request.get('max_workers')) is not int or request['max_workers'] != 1
            or request.get('restart') is not False):
        raise AdmissionError('Only one fixed worker, zero retries and no restart are admitted')
    if not re.fullmatch('[0-9a-f]{64}', str(request.get('recipe_sha256', ''))):
        raise AdmissionError('Frozen recipe hash required')
    if not isinstance(request.get('resource_spec'), dict) or not request['resource_spec']:
        raise AdmissionError('Immutable provider resource specification required')
    if quote.get('resource_spec') != request['resource_spec']:
        raise AdmissionError('Selected resources differ from priced resources')
    integer(quote.get('price_ceiling_until'), 'price_ceiling_until', 1)
    if quote['price_ceiling_until'] < now + request['runtime_seconds']:
        raise AdmissionError('Price ceiling does not cover maximum allocation lifetime')
    if not isinstance(quote.get('components'), dict) or set(quote['components']) != COMPONENTS:
        raise AdmissionError('All lifecycle cost components must be explicitly priced')
    for component in quote['components'].values():
        if not isinstance(component, dict) or set(component) != {'fixed', 'per_second'}:
            raise AdmissionError('Each cost component requires fixed and per_second microcurrency units')
        for key, value in component.items():
            integer(value, key)
    if not sum(c['per_second'] for c in quote['components'].values()):
        raise AdmissionError('Positive total lifetime cost rate required')


class BudgetLedger:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS policies (provider TEXT PRIMARY KEY, record TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, provider TEXT NOT NULL, record TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, job TEXT, record TEXT NOT NULL)')

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute('PRAGMA synchronous=FULL')
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def configure(self, provider, policy):
        with self.transaction() as db:
            if db.execute('SELECT 1 FROM jobs WHERE provider=?', (provider,)).fetchone():
                raise AdmissionError('Policy is immutable once commitments exist; reconcile before a reviewed migration')
            db.execute('INSERT OR REPLACE INTO policies VALUES (?, ?)', (provider, encoded(policy)))

    def refresh(self, provider, policy):
        """Refresh evidence conservatively; never replenish a credit epoch.

        Remaining billed balance can double-count settled costs here. This is
        intentional: a new credit epoch needs a separately reviewed migration,
        not an unattended reset that can release delayed liabilities.
        """
        policy = json.loads(encoded(policy))
        evidence(policy, int(self.clock()))
        with self.operation(), self.transaction() as db:
            result = db.execute('SELECT record FROM policies WHERE provider=?', (provider,)).fetchone()
            if not result:
                raise AdmissionError('Provider has no initial credit policy')
            old = json.loads(result[0])
            if policy['currency'] != old['currency'] or policy.get('credit_expires') != old['credit_expires']:
                raise AdmissionError('Currency or credit epoch change requires reviewed migration')
            integer(policy.get('allowance'), 'allowance')
            policy['allowance'] = min(old['allowance'], policy['allowance'])
            for key in ('production_reserve', 'safety_reserve', 'expiry_headroom'):
                integer(policy.get(key), key)
                policy[key] = max(old[key], policy[key])
            db.execute('UPDATE policies SET record=? WHERE provider=?', (encoded(policy), provider))
            db.execute('INSERT INTO events (job, record) VALUES (?, ?)', (None, encoded({'provider': provider, 'policy': policy})))

    def get(self, job):
        with self.transaction() as db:
            return self._get(db, job)

    @staticmethod
    def _get(db, job):
        row = db.execute('SELECT record FROM jobs WHERE id=?', (job,)).fetchone()
        if not row:
            raise AdmissionError('Unknown job')
        return json.loads(row[0])

    def _save(self, db, row):
        db.execute('UPDATE jobs SET record=? WHERE id=?', (encoded(row), row['id']))
        db.execute('INSERT INTO events (job, record) VALUES (?, ?)', (row['id'], encoded(row)))

    def _transition(self, row, **changes):
        row.update(changes)
        with self.transaction() as db:
            self._save(db, row)
        return row

    @contextmanager
    def operation(self):
        # One controller operation across every provider. Persist transitions
        # before external side effects; a crash drops this lock, not the claim.
        with self.path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise AdmissionError('Another controller operation is active') from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def launch(self, job, adapter, current_quote):
        with self.operation():
            row = self.get(job)
            if row['state'] != 'reserved' or adapter.name != row['provider']:
                raise AdmissionError('Existing launch or wrong provider; reconcile original identity')
            now = int(self.clock())
            with self.transaction() as db:
                policy = json.loads(db.execute('SELECT record FROM policies WHERE provider=?', (row['provider'],)).fetchone()[0])
            validate_bounds(policy, row['request'], current_quote, now)
            if (current_quote['components'] != row['quote']['components']
                    or now >= row['deadline']
                    or row['deadline'] > policy['credit_expires'] - policy['expiry_headroom']):
                raise AdmissionError('Price or available lifetime changed; original reservation is immutable')
            with self.transaction() as db:
                peers = [json.loads(r[0]) for r in db.execute('SELECT record FROM jobs WHERE provider=?', (row['provider'],))]
            if any(r.get('overrun_detected') or r['state'] in ('arming', 'submitting', 'unknown', 'terminating', 'termination_failed', 'overrun') for r in peers):
                raise AdmissionError('Uncertain provider commitment blocks replacement work')
            total = sum(max(r.get('confirmed_liability', 0), r.get('charged', r['reserved'])) for r in peers)
            if total + policy['production_reserve'] + policy['safety_reserve'] > policy['allowance']:
                raise AdmissionError('Refreshed allowance cannot cover existing commitments')
            inventory = adapter.inventory()
            known = {r['provider_id'] for r in peers if r['provider_id'] and r['state'] != 'settled'}
            resource_ids = inventory.get('resource_ids')
            if (inventory.get('complete') is not True or not isinstance(resource_ids, list)
                    or any(not isinstance(i, str) or not i for i in resource_ids)
                    or not set(resource_ids) <= known):
                raise AdmissionError('External/manual resources are not accounted for')
            self._transition(row, state='arming', launch_policy=policy)
            try:
                guard = adapter.arm(json.loads(encoded(row)))
                if (not isinstance(guard.get('provider_id'), str) or not guard['provider_id']
                        or guard['provider_id'] in known
                        or guard.get('deadline') != row['deadline']
                        or guard.get('verified') is not True
                        or guard.get('provisioning_bounded') is not True
                        or guard.get('descendants_bounded') is not True
                        or guard.get('resource_spec') != row['request']['resource_spec']
                        or not guard.get('mechanism')):
                    raise AdmissionError('Provider cannot enforce the complete allocation lifetime')
                self._transition(row, state='submitting', provider_id=guard['provider_id'], termination=guard)
                # Provisioning/guard setup consumes the same immutable lifetime.
                if int(self.clock()) >= row['deadline']:
                    raise AdmissionError('Deadline expired during setup')
                validate_bounds(policy, row['request'], current_quote, int(self.clock()))
                adapter.submit(json.loads(encoded(row)))
            except BaseException:
                self._transition(row, state='unknown')
                raise
            return self._transition(row, state='running')

    def reconcile(self, job, adapter, *, cancel=False):
        with self.operation():
            row = self.get(job)
            if adapter.name != row['provider'] or row['state'] == 'reserved':
                raise AdmissionError('Wrong provider or job was never submitted')
            if row['state'] == 'settled':
                return row
            if cancel:
                self._transition(row, cancel_requested=True)
            cleanup_due = row.get('cancel_requested') or int(self.clock()) >= row['deadline']
            try:
                # Cleanup by the already recorded identity must remain possible
                # when the provider's inventory/inspection endpoint is down.
                if cleanup_due:
                    if not row['provider_id']:
                        raise AdmissionError('Recover original provider identity before cleanup')
                    self._transition(row, state='terminating')
                    adapter.terminate(json.loads(encoded(row)))
                observed = adapter.inspect(json.loads(encoded(row)))
                if (observed.get('provider_id') != row['provider_id'] or not row['provider_id']
                        or not observed.get('evidence') or observed.get('currency') != row['policy']['currency']
                        or type(observed.get('terminal')) is not bool
                        or type(observed.get('residual_billable')) is not bool):
                    raise AdmissionError('Provider identity, currency or complete terminal evidence is missing')
                if cleanup_due and (not observed['terminal'] or observed['residual_billable']):
                    raise AdmissionError('Termination or residual resource cleanup remains unverified')
            except BaseException:
                self._transition(row, state='termination_failed' if row['state'] == 'terminating' else 'unknown')
                raise
            if not observed['terminal'] or observed['residual_billable']:
                return self._transition(row, state='overrun' if row.get('overrun_detected') else 'running', observation=observed)
            cost = observed.get('final_cost')
            if cost is None:
                return self._transition(row, state='overrun' if row.get('overrun_detected') else 'awaiting_billing', observation=observed,
                                        operational_result=observed.get('operational_result'))
            integer(cost, 'final_cost')
            if cost > row['reserved'] or row.get('overrun_detected'):
                self._transition(row, state='overrun', overrun_detected=True, observation=observed,
                                 confirmed_liability=max(cost, row.get('confirmed_liability', 0)))
                raise AdmissionError('Provider cost exceeded reservation; provider admission is blocked')
            return self._transition(row, state='settled', charged=cost, observation=observed,
                                    operational_result=observed.get('operational_result'))

    def cancel_reserved(self, job):
        """Release only a reservation that never entered provider arming."""
        with self.operation(), self.transaction() as db:
            row = self._get(db, job)
            if row['state'] != 'reserved':
                raise AdmissionError('Provider operation already started; reconcile original identity')
            row.update(state='settled', charged=0, operational_result='cancelled_before_arming')
            self._save(db, row)
            return row

    def record_scientific_verdict(self, job, verdict, artifact_sha256):
        if not isinstance(verdict, str) or not verdict.strip() or not re.fullmatch('[0-9a-f]{64}', str(artifact_sha256)):
            raise AdmissionError('Explicit scientific verdict and result artifact hash required')
        with self.operation(), self.transaction() as db:
            row = self._get(db, job)
            row['scientific_verdict'] = {'verdict': verdict, 'artifact_sha256': artifact_sha256}
            self._save(db, row)
            return row

    def recover_identity(self, job, adapter):
        """Read-only provider lookup after a lost guard response, never resubmit."""
        with self.operation():
            row = self.get(job)
            if adapter.name != row['provider'] or row['state'] not in ('arming', 'unknown') or row['provider_id']:
                raise AdmissionError('Identity recovery applies only to an unresolved guard operation')
            audit = adapter.lookup(json.loads(encoded(row)))
            if (audit.get('complete') is not True or not audit.get('evidence')
                    or audit.get('job_id') != row['id']
                    or audit.get('recipe_sha256') != row['request']['recipe_sha256']
                    or audit.get('resource_spec') != row['request']['resource_spec']
                    or audit.get('deadline') != row['deadline']
                    or not isinstance(audit.get('provider_id'), str) or not audit['provider_id']):
                raise AdmissionError('Provider lookup does not uniquely identify the original allocation')
            with self.transaction() as db:
                peers = [json.loads(r[0]) for r in db.execute('SELECT record FROM jobs WHERE provider=?', (row['provider'],))]
            if any(r['provider_id'] == audit['provider_id'] for r in peers):
                raise AdmissionError('Provider identity already belongs to a recorded commitment')
            return self._transition(row, state='unknown', provider_id=audit['provider_id'], identity_audit=audit)

    def reserve(self, job, provider, request, quote):
        # Serialize and decode caller-owned structures before persistence.
        request, quote = json.loads(encoded(request)), json.loads(encoded(quote))
        now = int(self.clock())
        with self.transaction() as db:
            saved = db.execute('SELECT record FROM policies WHERE provider=?', (provider,)).fetchone()
            if not saved:
                raise AdmissionError('Provider has no verified policy')
            policy = json.loads(saved[0])
            validate_bounds(policy, request, quote, now)
            fixed = sum(c['fixed'] for c in quote['components'].values())
            rate = sum(c['per_second'] for c in quote['components'].values())
            cost_lifetime = (request['max_cost'] - fixed) // rate
            lifetime = min(request['runtime_seconds'], cost_lifetime,
                           request['deadline'] - now,
                           policy['credit_expires'] - policy['expiry_headroom'] - now)
            if lifetime < request['runtime_seconds']:
                raise AdmissionError('Cannot cover complete bounded recipe lifetime')
            reserved = fixed + rate * lifetime
            rows = [json.loads(r[0]) for r in db.execute('SELECT record FROM jobs WHERE provider=?', (provider,))]
            committed = sum(max(r.get('confirmed_liability', 0), r.get('charged', r['reserved'])) for r in rows)
            available = policy['allowance'] - policy['production_reserve'] - policy['safety_reserve'] - committed
            if reserved > available:
                raise AdmissionError('Unreserved allowance is insufficient')
            row = dict(id=job, provider=provider, policy=policy, quote=quote, request=request,
                       estimate=reserved, reserved=reserved, deadline=now + lifetime,
                       created_at=now, state='reserved', provider_id=None,
                       termination=None, operational_result=None, scientific_verdict=None)
            try:
                db.execute('INSERT INTO jobs VALUES (?, ?, ?)', (job, provider, encoded(row)))
            except sqlite3.IntegrityError as error:
                raise AdmissionError('Job identity already exists; never repeat submission') from error
            self._save(db, row)
            return row

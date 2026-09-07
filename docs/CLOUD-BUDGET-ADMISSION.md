# Cloud admission and recovery boundary

The standard-library `sepalith.cloud_budget.BudgetLedger` implements durable
reservation and recovery logic. **Real cloud scheduling is not enabled.** There
is no Azure/Anyscale provider adapter or completed cloud lifecycle test for this
API. The earlier Azure smoke is not a financial protection mechanism.

The user reserves most cloud credit for a production-grade model. Preserve the
roughly USD 85 Anyscale production reserve. Treat remaining Azure credit as
production-reserved until an explicit bounded allowance is established. Current
credit balances, expiry and covered services must be verified independently;
reported historical balances are not default policies. Prefer local hardware and
free Kaggle capacity for eligible exploratory work.

## Implemented API

Use one absolute ledger path outside Git for every provider. SQLite immediate
transactions serialize reservations. A separate process lock serializes provider
operations; durable transitions precede external side effects. Keep the ledger
and its journals on a local filesystem with SQLite and flock support.

| Operation | Behavior |
|---|---|
| `configure(provider, policy)` | Establish an evidence-backed credit epoch before commitments exist |
| `refresh(provider, policy)` | Refresh evidence without increasing allowance, reducing reserves or changing currency/expiry |
| `reserve(id, provider, request, quote)` | Persist an immutable priced recipe, reserved cost and absolute deadline |
| `launch(id, adapter, quote)` | Recheck evidence, account inventory and guard; submit once only |
| `reconcile(id, adapter, cancel=False)` | Attempt due cleanup by recorded identity, verify terminal state and retain unsettled liabilities |
| `recover_identity(id, adapter)` | Recover a lost guard response through exact provider submission-key lookup; never resubmit |
| `cancel_reserved(id)` | Release only a reservation that never entered provider arming |
| `record_scientific_verdict(id, verdict, artifact_sha256)` | Record science independently of operational success and billing |
| `get(id)` | Return the persisted job, evidence, estimate, reservation, identity, deadline and outcomes |

Amounts are integer **millionths of the explicitly named USD or EUR currency**.
There is no foreign-exchange conversion. Price components must cover provisioning,
compute, evaluation, upload, shutdown, disks, IPs, storage, egress and taxes. Each
component provides a fixed amount plus a conservative per-second ceiling charged
over the entire allocation lifetime. Round fractional rates upward before input;
the evidence must explain units, tax treatment and service coverage. The quote
must price the exact immutable resource specification and bind its price ceiling
through the allocation deadline.

Admission evidence expires within 900 seconds. The allocation lifetime cannot
exceed the cost-derived bound, recipe bound or credit-expiry deadline with
headroom. Provisioning and cleanup belong inside that bound. Only one fixed
worker, zero retries and no restart loop are admitted in this first version.

Queued, running and billing-pending jobs retain reservations. Terminal state alone
does not release a reservation. Final billed cost remains debited after settlement.
Refresh intentionally can double-count already settled costs against a newly
reported remaining balance; it cannot replenish the credit epoch automatically.
An overrun preserves the greatest confirmed liability and blocks further provider
launches even when later billing observations omit it. Resolving that condition
requires a reviewed accounting migration, not another submission.

## Provider adapter requirements

Adapters are trusted resource-control implementations, not arbitrary shell
commands. They must use provider APIs and implement these operations:

- `inventory()` returns a complete list of billable resource identities across
  the account scope. Missing inventory or external/manual resources closes
  admission. An inventory snapshot cannot prevent unrelated concurrent spending.
- `arm(job)` establishes and verifies independent provider-side termination
  **before compute becomes possible**. Return the exact identity, immutable
  absolute deadline, mechanism and resource specification. Confirm that both
  provisioning and all descendants are bounded. Guest shutdown is insufficient.
  This operation must support idempotent lookup by the ledger job/recipe identity.
- `submit(job)` submits once with the verified guard, exact resource specification
  and no provider retries, autoscaling or restart loops. Any ambiguous result
  retains the commitment and blocks replacement work.
- `inspect(job)` identifies the allocation, terminal state, residual billable
  resources, billing currency and final cost when established. Retain evidence.
- `terminate(job)` idempotently deletes/deallocates the recorded resource and its
  descendants/residual billable resources. Keep results/checkpoints separately.
  Cleanup must work even when an inspection endpoint is unavailable.
- `lookup(job)` must uniquely recover the original guarded identity after a lost
  response, matching job ID, recipe hash, resource specification and deadline.

Every adapter call itself needs a bounded transport timeout. The local operation
lock is not a cloud watchdog; provider-side protection must survive workstation
failure. False adapter attestations defeat these guarantees. Raw paid provider
commands are not an alternative admission path for the queue owner.

## Validation and remaining work

The fake-provider suite covers concurrent admission, process restart, stale
credit/prices, currency/unit mistakes, resource-price mismatch, uncovered costs,
delayed billing, uncertain submission, lost guard identity, quota denial, stalled
setup, expired evidence, refreshed expiry headroom, deadline cleanup, inspection
outages, termination failure, residual billing, cancellation and persistent
overruns. Run it through `python3 scripts/check_core.py`.

Standards and spec reviews found five material gaps in the initial implementation;
regressions reproduced them and the fixes passed review. The reviews did not
approve a real provider adapter or certify a financial guarantee.

Remaining requirements are real provider adapters, enforced integration with all
cloud dispatch entry points, refreshed credit/price evidence, and one explicitly
budgeted cloud lifecycle test with verified cleanup. Azure PAYG protection remains
unverified under the user's credit-only constraint. Do not launch a paid test to
demonstrate this core while those requirements are unresolved.

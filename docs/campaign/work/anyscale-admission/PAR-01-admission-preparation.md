# PAR-01 Anyscale admission preparation

Observed 2026-09-12T12:41:28Z–12:44:00Z (UTC), using the existing
`/home/m0hawk/.local/bin/anyscale` authentication. This is a read-only
preparation record. No cloud, compute config, job, credential, data upload, or
model upload was created by this work.

## Current account and offer

- CLI version: `0.26.108`.
- `anyscale auth show` passed. The authenticated organization role is owner;
  identity values are intentionally omitted here.
- `anyscale cloud list --no-interactive --max-items 100 -o json` returned one
  default cloud: provider `AWS`, VM stack, region `us-east-2`.
- `anyscale cloud get/status --id cld_mh2xgqnvkq5squguqzf5wwwqzm
  --output-format json` both returned exit 1 with
  `Listing cloud resources is only supported for customer-hosted clouds.`
  This account cloud is therefore Anyscale-hosted; the `AWS` field identifies
  the provider backing its region, not a user BYOC AWS bill.
- The current provider credit endpoint returned:
  `current_balance_usd=95.433820846`, `amount_spent_usd=4.566179154`, and
  `total_granted_usd=100.0`. The active grant is `Free Trial` under
  `Self-Service`, effective `2026-08-30` through `2126-08-05`; expired,
  pending, and in-use commit lists were empty. This is live provider data and
  supersedes the earlier approximate USD95 user report while preserving that
  report in PRE-04 receipts.
- The public hosted price page currently publishes `NVIDIA A10G` at
  `1.3635 AC/hr` (the historical `g5.xlarge` has one A10G). It is a published
  list rate in Anyscale Credits, not an account-specific USD quote. The same
  page describes Hosted as Anyscale-managed infrastructure, Anyscale-hosted
  compute, and monthly credit-card invoicing. The hosted page does not attest
  that storage, transfer/egress, or every cleanup meter is covered by this
  grant. Keep the campaign all-in ceiling at USD60 and the allocation maximum
  at 18 hours until those residual meters are confirmed.
- Usage for `2026-08-30`–`2026-09-12` was `4.93108 AC` / `$4.566179` in the
  provider usage API. By instance type: `g5.xlarge` `$4.410546`,
  `g6.xlarge` `$0.151669`, and `m5.xlarge` `$0.003964`. These are historical
  usage estimates, not a current A10G quote. No instance-usage budgets or
  usage alerts were configured. `plan_status` reported
  `has_ever_had_a_plan=false`; active billing version was `1.9`.

The official references support these interpretations:

- [Anyscale pricing](https://www.anyscale.com/pricing) lists the hosted A10G
  class rate and the Hosted/BYOC billing and infrastructure distinction.
- [Anyscale clouds](https://docs.anyscale.com/clouds) describes serverless
  Anyscale-hosted clouds as fully managed and distinguishes them from AWS
  customer-defined clouds.
- [Credit history](https://docs.anyscale.com/administration/billing/credit-dashboard)
  defines current balance, spent, and granted credit fields.
- [Usage dashboard](https://docs.anyscale.com/administration/billing/usage-dashboard)
  warns that usage and displayed prices are estimates that can differ from a
  final bill.

## Jobs, capacity, and lifecycle evidence

The current commands below were read-only:

```text
/home/m0hawk/.local/bin/anyscale job list --v2 --cloud 'Anyscale Cloud' \
  -o json --no-interactive --max-items 100
/home/m0hawk/.local/bin/anyscale resource-quota list --cloud 'Anyscale Cloud' \
  --max-items 100 -o json
/home/m0hawk/.local/bin/anyscale compute-config list --cloud 'Anyscale Cloud' \
  --max-items 100 -o json
```

The job list contained 18 terminal jobs (9 `SUCCEEDED`, 9 `FAILED`) and no
`STARTING` or `RUNNING` job. Resource-quota and compute-config lists were
empty. The authenticated GPU snapshot API returned zero nodes and no instance
types at approximately `2026-09-12T12:40:09Z`; this means no active GPU was
visible, not that a future A10G allocation is available. The additional
instance-type API also returned an empty list. Current capacity remains
unknown.

The current status of the runbook-linked smoke ID is terminal `FAILED`, even
though the runbook records that historical attempt's 60-step training/loss
observations. Treat the runbook throughput/loss numbers as historical
capability evidence only. A separate historical successful status showed the
same fixed shape (`head_node: g5.xlarge`, `worker_nodes: []`, image
`anyscale/ray:2.57.0-py311`, `max_retries: 0`) and the existing entrypoint's
adapter push was configured; neither proves a new run or a current quote.

The current CLI and official jobs API support the controls needed for a future
lead-owned launch:

- `anyscale job submit -f <yaml> --max-retries 0 --timeout-s <seconds>`
  supports a per-run timeout;
- an inline compute config can pin one `head_node` and `worker_nodes: []`;
- `anyscale job status --id <id> -o json` is safe for polling;
- `anyscale job terminate --id <id>` stops the job and underlying cluster;
- jobs normally provision a cluster and terminate it after completion.

There is no CLI billing command, provider-side budget hard-stop, or tested
absolute deadline in this account. Budgets are documented as soft alerts and
do not terminate clusters. Lifecycle controls are documented/capable but not
tested in this read-only preparation.

## Concrete lead-owned preflight and shutdown recipe

Run this only after `theta0`, tokenizer/renderer, data manifest, source
snapshot, and a private artifact destination have passed the campaign gates.
Do not use the stale LFM/B7 template merely because it exists.

1. Build an immutable working directory from the exact reviewed source
   snapshot. Keep the generated YAML outside Git and inject `HF_TOKEN` only
   through the authorized secret/environment mechanism. Use one job config
   with `max_retries: 0`, `head_node.instance_type: g5.xlarge`, and
   `worker_nodes: []`; do not configure a job queue.
2. Set `LORA_REPO` to the authorized private model repository and a unique
   `RUN_NAME`. The reviewed `scripts/cloud/sft_entry.sh` uploads
   `final_lora` to `<RUN_NAME>/final_lora` before printing completion. Verify
   the private repository path and adapter hash after the job reaches a
   terminal success state. Node-local `/tmp` and local storage are ephemeral.
3. Compute a deadline guard before submission. Use a provider per-run timeout
   ending before the hard campaign stop, then retain a local watchdog for the
   hard stop:

```bash
STOP_EPOCH="$(date -u -d '2026-09-13T22:00:00Z' +%s)"
NOW_EPOCH="$(date -u +%s)"
RUN_TIMEOUT="$((STOP_EPOCH - NOW_EPOCH - 900))"  # 15-minute upload margin
test "$RUN_TIMEOUT" -gt 0

# Lead fills in the already-reviewed external YAML and submits once.
/home/m0hawk/.local/bin/anyscale job submit -f /path/to/private/job.yaml \
  --cloud 'Anyscale Cloud' --max-retries 0 --timeout-s "$RUN_TIMEOUT"
# Capture the returned prodjob ID; never parse or print secret env values.
JOB_ID='<returned-prodjob-id>'

# Run in a separate local supervisor process; status output is non-verbose.
while :; do
  STATE="$(/home/m0hawk/.local/bin/anyscale job status --id "$JOB_ID" \
    -o json | jq -r '.state')"
  case "$STATE" in
    SUCCEEDED|FAILED|TERMINATED|ERRORED|OUT_OF_RETRIES) break ;;
  esac
  NOW_EPOCH="$(date -u +%s)"
  if [ "$NOW_EPOCH" -ge "$STOP_EPOCH" ]; then
    /home/m0hawk/.local/bin/anyscale job terminate --id "$JOB_ID"
    break
  fi
  sleep 30
done
/home/m0hawk/.local/bin/anyscale job status --id "$JOB_ID" -o json
```

The provider timeout and local watchdog are complementary. A successful
status is required before treating the private adapter as present. At the
hard stop, invoke `job terminate` unconditionally if the job is not terminal,
poll until terminal, and reconcile usage/credit data. If the status or
termination API is unavailable, treat the lane as failed admission and do not
retry or submit a replacement. The current read-only checks do not establish
that this deadline path has actually terminated a live job.

## Admission decision

Preparation is **partial / launch blocked**. Authentication, current credit
balance/expiry, hosted cloud classification, historical fixed-node recipe,
zero-retry configuration, private adapter push path, and supported status/
terminate APIs are evidenced. Current A10G capacity, account-specific
all-in USD pricing, residual storage/transfer coverage, and independently
tested deadline termination are unresolved. Root/lead owns the final offer
coverage check and any launch; no local task depends on this optional lane.

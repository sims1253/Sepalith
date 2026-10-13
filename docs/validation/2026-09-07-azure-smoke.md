# Azure CPU compatibility smoke, 7 September 2026

One finite CPU program completed on Azure Container Instances in North Europe.
This validates basic remote execution, result collection and provenance; it does
not establish GPU access, training throughput or a scientific adoption result.

The user reports €171 remaining credit with a 30-day window and authorized a small
test. Modern GPU-family quotas were still zero in the inspected regions after the
pay-as-you-go upgrade. This test requested 1 vCPU and 1 GB memory, with no public
endpoint, credentials, mounted storage or persistent disk.

## Recipe and acceptance

The complete payload is [azure_cpu.py](../../scripts/cloud/probes/azure_cpu.py).
It uses Python's standard library to fit `y = 3x + 2` over 65 fixed points with
1,000 gradient steps. It writes and reloads a JSON checkpoint, requires mean
squared error below 1e-12, and prints the checkpoint, input/source hashes,
interpreter and execution details in a receipt. The code has a 60-second alarm.
Failure produces an explicit failed receipt and exits zero to avoid ACI's
possible restart of nonzero exits even with restart policy Never. Command exit
alone therefore never constitutes acceptance.

- Image: `docker.io/library/python@sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef`
- Source SHA-256: `9caf1d78cbdce131b918acdd94d4f450d1da49b02bb77e14034e43305bdc10d9`
- Input SHA-256: `00afa790bbdbe1280c5f15463d1ccf9380fb444f8f4a8eb375af2fb3240fea96`
- Checkpoint SHA-256: `16b4240ccdc62f207208bce561e54fae5d4d44b741c5c976c0eb86fd3cdecca2`
- Successful job: `sepalith-smoke-ddc40862d82d`, North Europe.
- Runtime: Python 3.12.14, `/usr/local/bin/python`, x86_64, one visible CPU affinity.
- Result: MSE 1.4126678364265504e-29; bias 2.0 and weight 3.0 within floating-point
  precision. Payload duration 0.014 seconds; this is not provisioned/billed time.
- Submission controller started 21:31:24 UTC; verified receipt at 21:32:06 UTC.

The controller journals before submission, polls the same resource, validates
receipt source and checkpoint hashes, and deletes its tagged resource group in
`finally`. A separate local watchdog triggers deletion after 12 minutes if cleanup
has not already completed. Provisioning/execution polling has a ten-minute limit.
The watchdog is independent of the controller but not of the workstation; it is
not an Azure-enforced spending cap. The finite remote process and Never restart
policy provide a second limit. No automatic job retries or follow-ons are enabled.
The exact controller and deployment bytes are archived with the run evidence.

## Failures retained

1. `sepalith-smoke-1c4c799d4e9d`, West Europe: Azure rejected creation because the
   region was not accepting new customers. No container allocation was accepted;
   the temporary resource group was deleted and absence verified.
2. `sepalith-smoke-b5cb8d5a86d0`, North Europe: creation accepted, but the local
   status reader assumed `instanceView` was a dictionary while Azure returned null
   during provisioning. The container and resource group were deleted, then their
   absence was verified before retry. Whether this payload executed is unknown;
   there is no collected success receipt, so it is not counted as a successful run.
3. The explicit final retry uses the tested null-safe
   [state reader](../../scripts/cloud/azure_container_state.py). Two regression
   tests cover pending/null responses and terminal exit-code preservation. All
   17 cloud-controller tests pass. Local payload checks also verified numerical
   convergence, checkpoint round-trip and a simulated disk-failure receipt.

## Cost and scope

Azure's retail-price API returned North Europe standard rates of €0.0348/vCPU-hour
and €0.0038/GB-hour, or €0.0386/hour for this allocation. West Europe's comparable
rate was €0.0444/hour. The test targeted less than €0.05; its brief deployments
suggest a fraction of a cent of compute, but billed cost has not yet been verified.
The user's €171 balance remains user-reported, not a refreshed billing receipt.
No claim of zero charge, guaranteed spending cap or GPU readiness follows.

Primary references: [ACI restart and billing semantics](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-restart-policy),
[pricing](https://azure.microsoft.com/en-us/pricing/details/container-instances/),
[retail-price API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).

## Cleanup and evidence

All three dedicated resource groups were confirmed absent after deletion. A final
subscription resource listing returned an empty list. No VM, container, disk or
public IP was left by this test. The [artifact index](2026-09-07-azure-smoke-artifacts.json)
contains private bucket references for source, deployment/controller bytes,
receipts, terminal state, cleanup confirmations and test output. Objects use
content-hash keys and were verified after downloading; the bucket itself is mutable.
Local originals remain under `~/.local/state/sepalith/azure-smoke-20260907*`.
The local experiment-runner migration and parked scientific authorizations are
unchanged by this cloud compatibility smoke.

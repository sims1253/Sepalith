# Budget admission review

Read-only review of the added `cloud_budget.py` and its tests, against the user
handoff of 8 September. The branch's pre-task fixed point was `6b59705`;
`aa0080f` and `267797d` integrated the existing runner migration. Standards and
spec reviewers worked independently. Real provider integration was explicitly
outside the completed implementation boundary.

## Standards

No documented standards violation was found. The core uses standard-library
dependencies, durable transitions, absolute state paths and separate scientific
verdicts, consistent with the existing runner and project map.

The reviewer found three correctness concerns:

1. Refreshed expiry headroom did not constrain an already reserved deadline.
2. An inspection outage prevented cleanup despite a recorded provider identity.
3. Credit or price evidence could expire during provider calls before submission.

All three were reproduced by regression tests and fixed. Launch now checks the
refreshed cutoff and revalidates evidence immediately before submission. Due
cleanup attempts recorded-identity termination before inspection and retains the
commitment when verification fails. The reviewer confirmed these fixes and found
no new material issue. Repeated query/decode logic was an optional duplication
heuristic, not a documented violation.

## Spec

The reviewer found three implementation gaps:

1. A later incomplete billing observation could erase a previously observed
   overrun and unblock launches using the original insufficient reservation.
2. Refreshed expiry headroom was not enforced at launch.
3. A reservation that never launched could not be cancelled or released.

Regressions reproduced these gaps. Overrun flags and greatest confirmed
liabilities now persist across incomplete observations. The refreshed cutoff is
enforced. `cancel_reserved()` atomically releases only commitments that never
entered arming. A separate scientific-verdict recording operation was also added.
The reviewer verified the fixes and found no further material implementation bug.

Real provider adapters, mandatory integration with cloud entry points and a tiny
cloud lifecycle test remain incomplete. Neither review authorizes cloud spending
or substitutes for those requirements.

Standards: three correctness findings resolved, no documented violations.
Spec: three implementation findings resolved; real-provider requirements remain.

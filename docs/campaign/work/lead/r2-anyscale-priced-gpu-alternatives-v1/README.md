# Anyscale priced GPU alternatives v1

This read-only review tests a distinct A100-40G route: one AWS
`p4d.24xlarge` node with eight physical GPUs. It is technically plausible for
the measured full-weight 16K footprint on one GPU, although the current trainer
does not shard and would leave seven GPUs unused.

The route is not ready for paid admission. Anyscale's public page gives an A100
category of 4.9591 AC/hour and a hosted dollar **from** price of $4.5916/hour.
It does not bind either number to an eight-GPU `p4d.24xlarge` node. The account's
read-only p4d-filtered fleet result contains no current records; this is neither
a catalog nor an entitlement failure. AWS confirms the exact shape and Ohio
support, and separately lists a $11.80/hour Capacity Block price, but that is an
AWS procurement product and cannot be charged to the current Anyscale credits.

The existing CLI token exposes API and signed billing-dashboard routes, not an
authenticated browser session or target-SKU quote endpoint. The remaining
prerequisite is an organization-owner export that binds the concrete hosted
p4d node, whole-node rate, credit eligibility, and enabled status. No cloud
resource was created or modified.

# PRE-04 long-context admission v2

This packet records a new, read-only route for obtaining account-specific
billing evidence for a larger hosted GPU. It does not allocate a job, create a
node, change a cloud, or authorize spend.

The prior packet exhausted the credit and historical-usage APIs, GPU-fleet
snapshots, saved compute configuration, resource quota, and plan/billing-version
surfaces. Those surfaces do not provide a complete hosted instance catalog, a
nonallocating entitlement check, or a target-shape quote. The v2 probe instead
uses the account billing portal and embedded billing dashboards, which were not
part of the prior cost packet.

The new route results are in `probe-evidence.json`:

- `GET /api/v2/organization_billing/manage_billing_url` returned a signed
  Stripe billing-portal URL. A one-time in-memory GET of that URL returned HTTP
  200 from `billing.stripe.com`. Only URL metadata and a body hash were stored;
  the signed URL and page body were discarded. The returned HTML contained
  invoice/subscription/rate markers, but no `g6e.2xlarge`, `p5.4xlarge`, L40S,
  or H100 identifier. This is an actionable account-owner route for inspecting
  or exporting a contract/invoice line item.
- `GET /api/v2/organization_billing/metronome_embedded_dashboard_url/{type}`
  returned signed URLs for `invoices`, `credits`, and `usage`. Each dashboard
  URL dereferenced to HTTP 200 at `embedded-dashboards.metronome.com`. The
  initial HTML is a client dashboard shell and contains no target shape or rate;
  the signed dashboard remains the account-visible place to inspect usage and
  invoices.
- `GET /api/v2/billing_scripts/{organization_id}` (contract info),
  `GET /api/v2/organization_billing/billing_versions?organization_id={organization_id}`, and
  `GET /api/v2/metronome_customer_info/{organization_id}` each returned HTTP
  403. The current token therefore cannot obtain the contract records directly.
  `GET /api/v2/organization_billing/alerts` returned an empty alert list.

No exact all-in Anyscale rate or target-shape enablement was verified. The
smallest missing account fact is one redacted export from the Stripe/Metronome
portal or an authorized contract endpoint that binds `g6e.2xlarge` (one L40S
48 GB) or `p5.4xlarge` (one H100 80 GB) to the existing hosted AWS `us-east-2`
cloud with: exact AC/USD hourly rate, billing increment, credit eligibility,
and enabled/launchable status. A zero target-node fleet snapshot remains
insufficient evidence of unavailability or future capacity.

The prior reconciliation binds current balance `$86.211311808` and campaign
headroom `$46.211311808` after release of the unused reservation. No rate is
invented and no number of fundable hours is derived until the target rate is
bound. The next safe action is to open the returned portal/dashboard under the
authorized account and export only the target SKU/contract fields, or have an
account billing administrator grant the read permission needed by the three
403 routes. A paid allocation probe remains a separate authorization gate.

Run the probe with the Anyscale CLI environment's Python:

```text
/home/m0hawk/.local/share/uv/tools/anyscale/bin/python \
  read_only_billing_route_probe.py \
  --dereference \
  --output probe-evidence.json
```

The `--dereference` option performs only HTTP GETs and persists no URL or
response body. Omit it when only API route status is required.

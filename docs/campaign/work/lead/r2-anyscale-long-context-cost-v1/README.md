# PRE-04 long-context cost review

This packet uses Anyscale's read-only credit and aggregated-usage APIs. It does
not query the GPU fleet menu again, create a compute configuration, allocate a
node, or contact the provider.

The four jobs covered by the $28 CPT envelope have accrued $7.883209310 in the
credit ledger since admission. The unused envelope is $20.116790690. The live
$86.211311808 balance already includes those debits, so subtracting $28 from it
would double count them.

The account has enough credit for another bounded experiment in principle.
Exact fundable time remains unknown: public hosted pricing has no L40S entry and
lists H100 as Contact Us, while SDK 0.26.108 has no read-only hosted entitlement
catalog or price validator for `g6e.2xlarge` or `p5.4xlarge`.

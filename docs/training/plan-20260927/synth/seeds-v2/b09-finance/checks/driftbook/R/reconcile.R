# skipped: needs dplyr/tidyr: clean_invoices, clean_payments, match_payments,
# reconciliation_summary, ageing_report, unbalanced_accounts

# extract_invoice_ref
refs <- c("PAYMENT INV-1001 THANKS", "INV1002", "REFUND 77", NA)
stopifnot(identical(extract_invoice_ref(refs), c("INV-1001", "INV-1002", NA, NA)))
stopifnot(identical(extract_invoice_ref(character(0)), character(0)))
stopifnot(identical(extract_invoice_ref("ref 4411", pattern = "[0-9]{4}"), "4411"))

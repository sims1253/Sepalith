make_invoices <- function() {
  tibble::tibble(
    invoice_id = c(" inv-1001", "INV-1002", "INV-1003", "INV-1004"),
    customer = c("acme ", "ACME", "Brightwell", "Corvid"),
    issued = as.Date(c("2026-01-05", "2026-02-10", "2026-03-01", "2026-03-20")),
    amount = c(100, 250.5, 80, 1200)
  )
}

make_payments <- function() {
  tibble::tibble(
    payment_id = c("P1", "P2", "P3", "P4"),
    reference = c("Payment INV-1001 thanks", "inv1002", "INV-1004 part", "refund 77"),
    received = as.Date(c("2026-01-20", "2026-03-01", "2026-04-02", "2026-04-03")),
    amount = c(100, 250.5, 1000, 15)
  )
}

test_that("clean_invoices trims and upper-cases identifiers", {
  out <- clean_invoices(make_invoices())
  expect_equal(out$invoice_id[1], "INV-1001")
  expect_equal(out$customer[1], "ACME")
  expect_s3_class(out$issued, "Date")
  expect_equal(nrow(out), 4)
})

test_that("clean_payments rounds amounts and drops missing ones", {
  pay <- make_payments()
  pay$amount <- c(100.004, NA, 1000, 15)

  out <- clean_payments(pay)
  expect_equal(nrow(out), 3)
  expect_equal(out$amount[1], 100)
  expect_equal(out$reference[2], "INV-1004 PART")
})

test_that("clean_invoices normalises the invoice prefix", {
  inv <- make_invoices()
  inv$invoice_id[2] <- "inv1002"
  out <- clean_invoices(inv)
  expect_equal(out$invoice_id[2], "INV-1002")
})

test_that("extract_invoice_ref normalises the separator", {
  refs <- c("PAYMENT INV-1001 THANKS", "INV1002", "REFUND 77", NA)
  expect_equal(extract_invoice_ref(refs), c("INV-1001", "INV-1002", NA, NA))
})

test_that("match_payments assigns every status", {
  out <- match_payments(make_invoices(), make_payments())
  # unallocated payments have no invoice id, so fall back to the payment id
  status <- stats::setNames(out$status, dplyr::coalesce(out$invoice_id, out$payment_id))

  expect_equal(unname(status[c("INV-1001", "INV-1002")]), c("matched", "matched"))
  expect_equal(unname(status["INV-1003"]), "unpaid")
  expect_equal(unname(status["INV-1004"]), "short")
  expect_equal(sum(out$status == "unallocated"), 1)
})

test_that("match_payments respects the tolerance", {
  pay <- make_payments()
  pay$amount[1] <- 100.4

  strict <- match_payments(make_invoices(), pay)
  loose <- match_payments(make_invoices(), pay, tolerance = 0.5)
  expect_equal(strict$status[strict$invoice_id %in% "INV-1001"], "over")
  expect_equal(loose$status[loose$invoice_id %in% "INV-1001"], "matched")
})

test_that("reconciliation_summary totals by status", {
  out <- reconciliation_summary(match_payments(make_invoices(), make_payments()))
  expect_setequal(out$status, c("matched", "unpaid", "short", "unallocated"))
  expect_equal(out$n[out$status == "matched"], 2)
  expect_equal(out$difference[out$status == "short"], -200)
})

test_that("ageing_report buckets open balances", {
  matched <- match_payments(make_invoices(), make_payments())
  out <- ageing_report(matched, as_of = as.Date("2026-04-30"))

  # Corvid paid 1000 of 1200, so only the shortfall is outstanding
  expect_equal(out$`31-60`[out$customer == "BRIGHTWELL"], 80)
  expect_equal(out$`31-60`[out$customer == "CORVID"], 200)
})

test_that("unbalanced_accounts reports only accounts out of balance", {
  ledger <- tibble::tibble(
    account = c("1200", "1200", "4000", "4000"),
    debit = c(500, NA, 120, 0),
    credit = c(NA, 500, 0, 100)
  )

  out <- unbalanced_accounts(ledger)
  expect_equal(out$account, "4000")
  expect_equal(out$imbalance, 20)
  expect_equal(nrow(unbalanced_accounts(ledger[1:2, ])), 0)
})

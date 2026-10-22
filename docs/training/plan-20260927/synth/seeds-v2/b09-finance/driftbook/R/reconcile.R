#' @title Standardise an invoice table
#'
#' @description Trims and upper-cases references, normalises the `INV-`
#'   prefix, coerces dates and rounds amounts to cents so that matching is
#'   not defeated by formatting noise.
#'
#' @param invoices A data frame of invoices with `invoice_id`, `customer`,
#'   `issued` and `amount` columns.
#' @return A tibble with the same columns, cleaned.
#' @export
clean_invoices <- function(invoices) {
  invoices %>%
    dplyr::mutate(
      # same normalisation as extract_invoice_ref(), so ids written as
      # "INV1002" still meet payments that quote "INV-1002"
      invoice_id = sub("^INV-?", "INV-", toupper(trimws(.data$invoice_id))),
      customer = toupper(trimws(.data$customer)),
      issued = as.Date(.data$issued),
      amount = round(as.numeric(.data$amount), 2)
    ) %>%
    dplyr::filter(!is.na(.data$invoice_id), .data$invoice_id != "")
}

#' @title Standardise a payments table
#'
#' @description Trims and upper-cases references, coerces dates and rounds
#'   amounts to cents so that matching is not defeated by formatting noise.
#'
#' @param payments A data frame of bank payments with `payment_id`,
#'   `reference`, `received` and `amount` columns.
#' @return A tibble with the same columns, cleaned.
#' @export
clean_payments <- function(payments) {
  payments %>%
    dplyr::mutate(
      payment_id = trimws(.data$payment_id),
      reference = toupper(trimws(.data$reference)),
      received = as.Date(.data$received),
      amount = round(as.numeric(.data$amount), 2)
    ) %>%
    dplyr::filter(!is.na(.data$amount))
}

#' @title Extract invoice numbers from free-text payment references
#'
#' @param reference Character vector of payment references.
#' @param pattern Regular expression matching an invoice number.
#' @return A character vector with the first match, or `NA`.
#' @export
extract_invoice_ref <- function(reference, pattern = "INV-?[0-9]{4,8}") {
  matched <- regexpr(pattern, reference)
  hit <- regmatches(reference, matched)
  out <- rep(NA_character_, length(reference))
  # regmatches() drops non-matches, so write the hits back by position
  out[!is.na(matched) & matched > 0] <- hit

  sub("^INV-?", "INV-", out)
}

#' @title Match payments to invoices
#'
#' @description Matches first on the invoice number found in the payment
#'   reference and then classifies each pair by how far the amounts differ.
#'
#' @param invoices A data frame of invoices with `invoice_id`, `customer`,
#'   `issued` and `amount` columns.
#' @param payments A data frame of bank payments with `payment_id`,
#'   `reference`, `received` and `amount` columns.
#' @param tolerance Absolute amount difference still treated as a match.
#' @return A tibble with one row per invoice-payment pair and a `status`
#'   column of `"matched"`, `"short"`, `"over"`, `"unpaid"` or
#'   `"unallocated"`.
#' @export
match_payments <- function(invoices, payments, tolerance = 0.01) {
  inv <- clean_invoices(invoices)
  pay <- clean_payments(payments) %>%
    dplyr::mutate(invoice_id = extract_invoice_ref(.data$reference))

  # a full join keeps both open invoices and payments we could not place
  dplyr::full_join(inv, pay, by = "invoice_id", suffix = c("_invoiced", "_paid")) %>%
    dplyr::mutate(
      difference = .data$amount_paid - .data$amount_invoiced,
      status = dplyr::case_when(
        is.na(.data$payment_id) ~ "unpaid",
        is.na(.data$amount_invoiced) ~ "unallocated",
        abs(.data$difference) <= tolerance ~ "matched",
        .data$difference < 0 ~ "short",
        TRUE ~ "over"
      ),
      days_to_pay = as.integer(.data$received - .data$issued)
    )
}

#' @title Summarise reconciliation results by status
#'
#' @param matched A tibble returned by [match_payments()].
#' @return A tibble with counts and totals per status.
#' @family reconciliation reports
#' @export
reconciliation_summary <- function(matched) {
  matched %>%
    dplyr::group_by(.data$status) %>%
    dplyr::summarise(
      n = dplyr::n(),
      invoiced = sum(.data$amount_invoiced, na.rm = TRUE),
      paid = sum(.data$amount_paid, na.rm = TRUE),
      difference = sum(.data$difference, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(.data$n))
}

#' @title Ageing report of open invoices
#'
#' @description Buckets unpaid and short-paid invoices by days outstanding
#'   at `as_of`.
#'
#' @param matched A tibble returned by [match_payments()].
#' @param as_of Date at which outstanding balances are measured.
#' @param breaks Upper bounds, in days, of the ageing buckets.
#' @return A tibble with outstanding totals per customer and bucket.
#' @family reconciliation reports
#' @export
ageing_report <- function(matched, as_of = Sys.Date(), breaks = c(30, 60, 90)) {
  breaks <- sort(breaks)
  k <- length(breaks)
  # middle buckets only exist when there are at least two breaks
  middle <- if (k > 1) paste0(breaks[-k] + 1, "-", breaks[-1]) else character(0)
  labels <- c(paste0("0-", breaks[1]), middle, paste0(breaks[k] + 1, "+"))

  matched %>%
    dplyr::filter(.data$status %in% c("unpaid", "short")) %>%
    dplyr::mutate(
      outstanding = .data$amount_invoiced - dplyr::coalesce(.data$amount_paid, 0),
      age = as.integer(as.Date(as_of) - .data$issued),
      bucket = cut(.data$age, c(-Inf, breaks, Inf), labels = labels)
    ) %>%
    dplyr::group_by(.data$customer, .data$bucket) %>%
    dplyr::summarise(outstanding = sum(.data$outstanding), .groups = "drop") %>%
    tidyr::pivot_wider(names_from = "bucket", values_from = "outstanding",
      values_fill = 0)
}

#' @title Check that a ledger balances
#'
#' @description Compares the sum of debits and credits per account and
#'   reports any account whose difference exceeds `tolerance`.
#'
#' @param ledger A data frame of journal lines with `account`, `debit` and
#'   `credit` columns.
#' @param tolerance Absolute amount difference still treated as a match.
#' @return A tibble of unbalanced accounts; empty when everything balances.
#' @export
unbalanced_accounts <- function(ledger, tolerance = 0.01) {
  ledger %>%
    dplyr::mutate(dplyr::across(c("debit", "credit"), ~ dplyr::coalesce(.x, 0))) %>%
    dplyr::group_by(.data$account) %>%
    dplyr::summarise(
      debit = sum(.data$debit),
      credit = sum(.data$credit),
      .groups = "drop"
    ) %>%
    dplyr::mutate(imbalance = round(.data$debit - .data$credit, 2)) %>%
    dplyr::filter(abs(.data$imbalance) > tolerance)
}

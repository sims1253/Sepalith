#' Build a cash flow statement with the indirect method
#'
#' The statement has one line per item, in this order: `net.income`,
#' `depreciation`, `change.receivables`, `change.inventory`,
#' `change.payables`, `operating`, `capex`, `investing`, `debt.change`,
#' `dividends`, `financing` and `net.change`.
#'
#' @param is An income statement with lines `net.income`.
#' @param current,prior Balance sheets with lines `receivables`,
#'   `inventory`, `payables`, `short.term.debt` and `long.term.debt`.
#' @param depreciation Depreciation charged in the period.
#' @param capex Capital expenditure in the period, as a positive number.
#' @param dividends Dividends paid in the period, as a positive number.
#' @return A named numeric vector of class `fin.statement`.
#' @export
build.cash.flow <- function(is, current, prior, depreciation, capex, dividends) {
  cf <- c()

  # Operating activities
  cf["net.income"] <- is[["net.income"]]
  cf["depreciation"] <- depreciation
  cf["change.receivables"] <- -(current[["receivables"]] - prior[["receivables"]])
  cf["change.inventory"] <- -(current[["inventory"]] - prior[["inventory"]])
  cf["change.payables"] <- current[["payables"]] - prior[["payables"]]
  cf["operating"] <- sum(cf[c("net.income", "depreciation", "change.receivables",
    "change.inventory", "change.payables")])
  # Investing activities
  cf["capex"] <- -capex
  cf["investing"] <- cf[["capex"]]
  # Financing activities
  debt.now <- current[["short.term.debt"]] + current[["long.term.debt"]]
  debt.before <- prior[["short.term.debt"]] + prior[["long.term.debt"]]
  cf["debt.change"] <- debt.now - debt.before
  cf["dividends"] <- -dividends
  cf["financing"] <- cf[["debt.change"]] + cf[["dividends"]]

  cf["net.change"] <- cf[["operating"]] + cf[["investing"]] + cf[["financing"]]
  structure(cf, class = c("fin.statement", "numeric"))
}

#' Reconcile a cash flow statement with the balance sheets
#'
#' @param cf A cash flow statement as returned by [build.cash.flow()].
#' @param current,prior Balance sheets with line `cash`.
#' @param tolerance Largest absolute difference treated as zero.
#' @return A list with elements `opening`, `closing`, `net.change`,
#'   `difference` and `ok`.
#' @export
reconcile.cash <- function(cf, current, prior, tolerance = 0.005) {
  opening <- prior[["cash"]]
  closing <- current[["cash"]]
  difference <- (closing - opening) - cf[["net.change"]]
  list(
    opening = opening,
    closing = closing,
    net.change = cf[["net.change"]],
    difference = difference,
    ok = abs(difference) <= tolerance
  )
}

#' Free cash flow under different definitions
#'
#' @details
#' * `"simple"`: `operating` plus `capex`;
#' * `"firm"`: as `"simple"`, with after-tax interest added back;
#' * `"equity"`: as `"simple"`, plus `debt.change`;
#' * `"levered"`: as `"equity"`, with after-tax interest deducted;
#' * `"owner"`: `net.income` plus `depreciation` plus `capex`.
#'
#' @param cf A cash flow statement as returned by [build.cash.flow()].
#' @param type One of `"simple"`, `"firm"`, `"equity"`, `"levered"` or
#'   `"owner"`.
#' @param interest Interest expense of the period, used by `"firm"`.
#' @param tax.rate Tax rate as a decimal, used by `"firm"`.
#' @return A single number.
#' @export
free.cash.flow <- function(cf, type = c("simple", "firm", "equity", "levered", "owner"),
                           interest = 0, tax.rate = 0) {
  type <- match.arg(type)
  switch(type,
    simple = {
      cf[["operating"]] + cf[["capex"]]
    },
    firm = {
      base <- cf[["operating"]] + cf[["capex"]]
      base + interest * (1 - tax.rate)
    },
    equity = {
      base <- cf[["operating"]] + cf[["capex"]]
      base + cf[["debt.change"]]
    },
    levered = {
      base <- cf[["operating"]] + cf[["capex"]] + cf[["debt.change"]]
      base - interest * (1 - tax.rate)
    },
    owner = {
      cf[["net.income"]] + cf[["depreciation"]] + cf[["capex"]]
    }
  )
}

#' Build an operating cash flow statement with the direct method
#'
#' Receipts and payments are grouped by `category`. The statement has one
#' line per category, in this order: `customers`, `suppliers`,
#' `employees`, `interest` and `taxes`, followed by `operating`.
#'
#' @param transactions A data frame of bank transactions with columns
#'   `category` and `amount` (receipts positive, payments negative).
#' @return A named numeric vector of class `fin.statement`.
#' @export
direct.cash.flow <- function(transactions) {
  cf <- c()
  amount <- transactions$amount
  category <- transactions$category

  # Receipts from customers
  cf["customers"] <- sum(amount[category == "customers"])
  # Payments to suppliers
  cf["suppliers"] <- sum(amount[category == "suppliers"])
  # Payments to employees
  cf["employees"] <- sum(amount[category == "employees"])
  # Payments of interest
  cf["interest"] <- sum(amount[category == "interest"])
  # Payments of taxes
  cf["taxes"] <- sum(amount[category == "taxes"])

  unknown <- setdiff(unique(category), names(cf))
  if (length(unknown) > 0L) {
    warning("Ignored transactions in categories: ", paste(unknown, collapse = ", "),
      call. = FALSE)
  }
  cf["operating"] <- sum(cf)
  structure(cf, class = c("fin.statement", "numeric"))
}

#' Project a running cash balance
#'
#' The low point is compared with `floor` and the high point with
#' `ceiling`, so both a funding need and surplus cash to invest are
#' visible.
#'
#' @param opening Opening cash balance.
#' @param inflows,outflows Numeric vectors of equal length with the
#'   expected receipts and payments per period, both as positive numbers.
#' @param floor Minimum acceptable balance.
#' @param ceiling Balance above which cash should be invested.
#' @return A list with elements `balance` (closing balance per period),
#'   `lowest` and `highest` (each a list with `period`, `value` and
#'   `gap`, the distance to `floor` or `ceiling`), `below` and `above`
#'   (periods outside the band).
#' @export
project.cash <- function(opening, inflows, outflows, floor = 0, ceiling = Inf) {
  if (length(inflows) != length(outflows)) {
    stop("`inflows` and `outflows` must have the same length.", call. = FALSE)
  }
  if (floor > ceiling) {
    stop("`floor` must not exceed `ceiling`.", call. = FALSE)
  }
  balance <- opening + cumsum(inflows - outflows)

  # Track the low point against the floor
  low <- which.min(balance)
  lowest <- list(period = low, value = balance[low], gap = balance[low] - floor)
  # Track the high point against the ceiling
  high <- which.max(balance)
  highest <- list(period = high, value = balance[high], gap = ceiling - balance[high])

  list(
    balance = balance,
    lowest = lowest,
    highest = highest,
    below = which(balance < floor),
    above = which(balance > ceiling)
  )
}

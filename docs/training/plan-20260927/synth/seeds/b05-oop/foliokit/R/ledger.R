#' Transaction types understood by a ledger
#'
#' @return A character vector.
#' @keywords internal
ledger_types <- function() {
  c("deposit", "withdrawal", "buy", "sell", "dividend", "fee")
}

#' Construct a ledger
#'
#' Low-level constructor for the `foliokit_ledger` class. Assumes its input
#' is already well-formed; use [ledger()] for user-facing input.
#'
#' @param entries A data frame with columns `date`, `type`, `symbol`,
#'   `quantity`, `price` and `amount`.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_ledger` object.
#' @keywords internal
new_ledger <- function(entries, currency = "USD") {
  stopifnot(is.data.frame(entries))
  structure(
    list(entries = entries, currency = currency),
    class = "foliokit_ledger"
  )
}

#' Validate a ledger
#'
#' Checks column presence and types, transaction types, and that trades
#' carry a symbol and a positive quantity.
#'
#' @param x A `foliokit_ledger` object.
#' @return `x`, invisibly, if it is valid; otherwise an error.
#' @keywords internal
validate_ledger <- function(x) {
  e <- x$entries
  required <- c("date", "type", "symbol", "quantity", "price", "amount")
  missing_cols <- setdiff(required, names(e))
  if (length(missing_cols) > 0L) {
    stop("Ledger is missing column(s): ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  if (!inherits(e$date, "Date")) {
    stop("Column `date` must be of class Date.", call. = FALSE)
  }
  bad_type <- setdiff(unique(e$type), ledger_types())
  if (length(bad_type) > 0L) {
    stop("Unknown transaction type(s): ", paste(bad_type, collapse = ", "), call. = FALSE)
  }
  is_trade <- e$type %in% c("buy", "sell")
  if (any(is_trade & (is.na(e$symbol) | !(e$quantity > 0)))) {
    stop("Every buy and sell needs a symbol and a positive quantity.", call. = FALSE)
  }
  if (is.unsorted(e$date)) {
    stop("Ledger entries must be in date order.", call. = FALSE)
  }
  invisible(x)
}

#' Create a ledger
#'
#' Builds a ledger from vectors of transactions. Cash amounts are derived
#' from quantity and price for trades, so only cash movements need `amount`.
#'
#' @param date A vector of transaction dates, coercible with [as.Date()].
#' @param type A character vector of transaction types.
#' @param symbol A character vector of instrument symbols; `NA` for cash.
#' @param quantity A numeric vector of units traded.
#' @param price A numeric vector of unit prices.
#' @param amount A numeric vector of cash amounts for non-trade entries.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_ledger` object.
#' @export
ledger <- function(date = as.Date(character()), type = character(), symbol = NA_character_,
                   quantity = NA_real_, price = NA_real_, amount = NA_real_,
                   currency = "USD") {
  n <- length(type)
  entries <- data.frame(
    date = as.Date(date),
    type = tolower(as.character(type)),
    symbol = rep_len(as.character(symbol), n),
    quantity = rep_len(as.numeric(quantity), n),
    price = rep_len(as.numeric(price), n),
    amount = rep_len(as.numeric(amount), n),
    stringsAsFactors = FALSE
  )
  sign <- ifelse(entries$type == "buy", -1, 1)
  is_trade <- entries$type %in% c("buy", "sell")
  entries$amount[is_trade] <- sign[is_trade] * entries$quantity[is_trade] * entries$price[is_trade]
  outflow <- entries$type %in% c("withdrawal", "fee")
  entries$amount[outflow] <- -abs(entries$amount[outflow])
  entries <- entries[order(entries$date), , drop = FALSE]
  rownames(entries) <- NULL
  validate_ledger(new_ledger(entries, toupper(currency)))
}

#' Append transactions to a ledger
#'
#' @param x A `foliokit_ledger` object.
#' @param ... Arguments passed to [ledger()] describing the new entries.
#' @return A new `foliokit_ledger` object containing all entries.
#' @export
ledger_add <- function(x, ...) {
  addition <- ledger(..., currency = x$currency)
  combined <- rbind(x$entries, addition$entries)
  combined <- combined[order(combined$date), , drop = FALSE]
  rownames(combined) <- NULL
  validate_ledger(new_ledger(combined, x$currency))
}

#' Format a ledger
#'
#' @param x A `foliokit_ledger` object.
#' @param n Maximum number of entries to include.
#' @param ... Ignored.
#' @return A character vector, one element per line.
#' @export
format.foliokit_ledger <- function(x, n = 10L, ...) {
  e <- x$entries
  header <- sprintf("<foliokit_ledger> %d entries in %s", nrow(e), x$currency)
  if (nrow(e) == 0L) {
    return(header)
  }
  shown <- utils::head(e, n)
  lines <- sprintf(
    "%s  %-10s %-6s %12s",
    format(shown$date), shown$type, ifelse(is.na(shown$symbol), "", shown$symbol),
    format(money(shown$amount, x$currency))
  )
  if (nrow(e) > n) {
    lines <- c(lines, sprintf("# ... with %d more entries", nrow(e) - n))
  }
  c(header, lines)
}

#' Print a ledger
#'
#' @param x A `foliokit_ledger` object.
#' @param n Maximum number of entries to include.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.foliokit_ledger <- function(x, n = 10L, ...) {
  cat(format(x, n = n), sep = "\n")
  invisible(x)
}

#' Summarise a ledger
#'
#' Totals the cash flows of a ledger by transaction type and reports the
#' date range covered.
#'
#' @param object A `foliokit_ledger` object.
#' @param ... Ignored.
#' @return A `summary_foliokit_ledger` object.
#' @export
summary.foliokit_ledger <- function(object, ...) {
  e <- object$entries
  totals <- tapply(e$amount, factor(e$type, levels = ledger_types()), sum)
  totals[is.na(totals)] <- 0
  structure(
    list(
      n = nrow(e),
      from = if (nrow(e) > 0L) min(e$date) else NA,
      to = if (nrow(e) > 0L) max(e$date) else NA,
      totals = totals,
      cash = sum(e$amount),
      currency = object$currency
    ),
    class = "summary_foliokit_ledger"
  )
}

#' Print a ledger summary
#'
#' @param x A `summary_foliokit_ledger` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.summary_foliokit_ledger <- function(x, ...) {
  cat("Ledger summary (", x$currency, ")\n", sep = "")
  cat("  entries: ", x$n, "\n", sep = "")
  if (x$n > 0L) {
    cat("  period:  ", format(x$from), " to ", format(x$to), "\n", sep = "")
  }
  for (type in names(x$totals)) {
    cat("  ", format(type, width = 11), format(money(x$totals[[type]], x$currency)), "\n",
      sep = "")
  }
  cat("  cash:    ", format(money(x$cash, x$currency)), "\n", sep = "")
  invisible(x)
}

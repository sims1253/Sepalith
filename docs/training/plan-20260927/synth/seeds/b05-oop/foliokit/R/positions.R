#' Construct a positions table
#'
#' Low-level constructor for the `foliokit_positions` class, a data frame of
#' holdings with one row per symbol.
#'
#' @param holdings A data frame with columns `symbol`, `quantity` and `cost`.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_positions` object.
#' @keywords internal
new_positions = function(holdings, currency = "USD") {
  stopifnot(is.data.frame(holdings))
  structure(
    holdings,
    currency = currency,
    class = c("foliokit_positions", "data.frame")
  )
}

#' Validate a positions table
#'
#' @param x A `foliokit_positions` object.
#' @return `x`, invisibly, if it is valid; otherwise an error.
#' @keywords internal
validate_positions = function(x) {
  if (!all(c("symbol", "quantity", "cost") %in% names(x))) {
    stop("Positions need `symbol`, `quantity` and `cost` columns.", call. = FALSE)
  }
  if (anyDuplicated(x$symbol) > 0L) {
    stop("Each symbol may appear only once in a positions table.", call. = FALSE)
  }
  if (any(x$quantity < 0)) {
    short = x$symbol[x$quantity < 0]
    stop("Short positions are not supported: ", paste(short, collapse = ", "), call. = FALSE)
  }
  invisible(x)
}

#' Derive positions from a ledger
#'
#' Replays the trades in a ledger to find the quantity held and the average
#' cost of each symbol. Sells reduce the cost basis proportionally.
#'
#' @param x A `foliokit_ledger` object.
#' @param as_of Only entries on or before this date are used.
#' @return A `foliokit_positions` object.
#' @export
ledger_positions = function(x, as_of = Sys.Date()) {
  trades = x$entries |>
    subset(type %in% c("buy", "sell") & date <= as.Date(as_of))
  symbols = sort(unique(trades$symbol))
  qty = stats::setNames(numeric(length(symbols)), symbols)
  cost = qty
  for (i in seq_len(nrow(trades))) {
    s = trades$symbol[[i]]
    q = trades$quantity[[i]]
    if (trades$type[[i]] == "buy") {
      qty[[s]] = qty[[s]] + q
      cost[[s]] = cost[[s]] + q * trades$price[[i]]
    } else {
      if (q > qty[[s]]) {
        stop("Sell of ", q, " ", s, " exceeds the ", qty[[s]], " held.", call. = FALSE)
      }
      cost[[s]] = cost[[s]] * (1 - q / qty[[s]])
      qty[[s]] = qty[[s]] - q
    }
  }
  held = qty > 0
  holdings = data.frame(
    symbol = symbols[held],
    quantity = unname(qty[held]),
    cost = unname(cost[held]),
    stringsAsFactors = FALSE
  )
  new_positions(holdings, x$currency) |> validate_positions()
}

#' Value positions at market prices
#'
#' @param x A `foliokit_positions` object.
#' @param prices A named numeric vector of prices, one per symbol.
#' @return `x` with added columns `price`, `value`, `pnl` and `weight`.
#' @export
value_positions = function(x, prices) {
  missing_px = setdiff(x$symbol, names(prices))
  if (length(missing_px) > 0L) {
    stop("No price for: ", paste(missing_px, collapse = ", "), call. = FALSE)
  }
  x$price = unname(prices[x$symbol])
  x$value = x$quantity * x$price
  x$pnl = x$value - x$cost
  x$weight = if (sum(x$value) > 0) x$value / sum(x$value) else rep(NA_real_, nrow(x))
  x
}

#' Convert positions to a plain data frame
#'
#' @param x A `foliokit_positions` object.
#' @param row.names Ignored; positions never carry row names.
#' @param optional Ignored.
#' @param ... Ignored.
#' @return A data frame with a `currency` column added.
#' @export
as.data.frame.foliokit_positions = function(x, row.names = NULL, optional = FALSE, ...) {
  currency = attr(x, "currency")
  attr(x, "currency") = NULL
  class(x) = "data.frame"
  x$currency = rep(currency, nrow(x))
  rownames(x) = NULL
  x
}

#' Format a positions table
#'
#' @param x A `foliokit_positions` object.
#' @param ... Ignored.
#' @return A character vector, one element per line.
#' @export
format.foliokit_positions = function(x, ...) {
  currency = attr(x, "currency")
  header = sprintf("<foliokit_positions> %d holding(s) in %s", nrow(x), currency)
  if (nrow(x) == 0L) {
    return(header)
  }
  cost = format(money(x$cost, currency))
  body = sprintf("%-8s %10s  cost %s", x$symbol, format(x$quantity, big.mark = ","), cost)
  if ("value" %in% names(x)) {
    value = format(money(x$value, currency))
    body = paste0(body, sprintf("  value %s (%5.1f%%)", value, 100 * x$weight))
  }
  c(header, body)
}

#' Print a positions table
#'
#' @param x A `foliokit_positions` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.foliokit_positions = function(x, ...) {
  cat(format(x), sep = "\n")
  invisible(x)
}

#' Monthly net cash flow of a ledger
#'
#' @param x A `foliokit_ledger` object.
#' @param types Transaction types to include.
#' @return A data frame with columns `month` and `net`.
#' @export
ledger_cash_by_month = function(x, types = c("deposit", "withdrawal", "dividend", "fee")) {
  types = match.arg(types, ledger_types(), several.ok = TRUE)
  flows = x$entries |>
    subset(type %in% types)
  if (nrow(flows) == 0L) {
    return(data.frame(month = character(), net = numeric()))
  }
  flows$month = format(flows$date, "%Y-%m")
  stats::aggregate(amount ~ month, data = flows, FUN = sum) |>
    stats::setNames(c("month", "net"))
}

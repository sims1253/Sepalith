#' Display conventions for a currency
#'
#' Symbols are kept to ASCII so the output is safe in every table format.
#'
#' @param code One of `"EUR"`, `"USD"`, `"GBP"`, `"JPY"` or `"CHF"`.
#' @return A list with elements `symbol`, `sep`, `position`, `digits`,
#'   `big_mark` and `decimal_mark`.
#' @keywords internal
currency_spec <- function(code = c("EUR", "USD", "GBP", "JPY", "CHF")) {
  code <- match.arg(code)
  switch(code,
    EUR = list(
      symbol = "EUR",
      sep = " ",
      position = "suffix",
      digits = 2L,
      big_mark = ".",
      decimal_mark = ","
    ),
    USD = list(
      symbol = "$",
      sep = "",
      position = "prefix",
      digits = 2L,
      big_mark = ",",
      decimal_mark = "."
    ),
    GBP = list(
      symbol = "GBP",
      sep = " ",
      position = "prefix",
      digits = 2L,
      big_mark = ",",
      decimal_mark = "."
    ),
    JPY = list(
      symbol = "JPY",
      sep = " ",
      position = "prefix",
      digits = 0L,
      big_mark = ",",
      decimal_mark = "."
    ),
    CHF = list(
      symbol = "CHF",
      sep = " ",
      position = "prefix",
      digits = 2L,
      big_mark = "'",
      decimal_mark = "."
    )
  )
}

#' Validate the shared arguments of the currency helpers
#'
#' @param x A numeric vector of amounts.
#' @param code A single currency code.
#' @param rates A named numeric vector of exchange rates per unit of `EUR`.
#' @return `TRUE`, invisibly, when every argument is valid.
#' @keywords internal
check_currency_args <- function(x, code, rates) {
  # Validate `x`
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  # Validate `code`
  if (!is.character(code) || length(code) != 1L || is.na(code)) {
    stop("`code` must be a single string.", call. = FALSE)
  }
  # Validate `rates`
  if (!is.numeric(rates) || is.null(names(rates))) {
    stop("`rates` must be a named numeric vector.", call. = FALSE)
  }

  if (any(is.na(rates) | rates <= 0)) {
    stop("`rates` must be positive.", call. = FALSE)
  }

  invisible(TRUE)
}

#' Round amounts using a named rule
#'
#' @param x A numeric vector of amounts.
#' @param digits Number of decimal places to keep.
#' @param method One of `"half_up"`, `"half_even"`, `"down"` or `"up"`;
#'   `"down"` and `"up"` move towards and away from zero.
#' @return A numeric vector the same length as `x`.
#' @export
round_amount <- function(x, digits = 2L, method = c("half_up", "half_even", "down", "up")) {
  method <- match.arg(method)
  factor <- 10^digits
  switch(method,
    half_up = {
      sign(x) * floor(abs(x) * factor + 0.5) / factor
    },
    half_even = {
      round(x, digits)
    },
    down = {
      sign(x) * floor(abs(x) * factor) / factor
    },
    up = {
      sign(x) * ceiling(abs(x) * factor) / factor
    }
  )
}

#' Format monetary amounts
#'
#' Negative amounts get a leading minus, or parentheses when `accounting`
#' is `TRUE`.
#'
#' @param x A numeric vector of amounts.
#' @param code A single currency code.
#' @param accounting Whether to wrap negative amounts in parentheses.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family currency formatters
#' @export
fmt_currency <- function(x, code = "EUR", accounting = FALSE, na_string = "") {
  spec <- currency_spec(code)
  body <- formatC(
    abs(x),
    format = "f",
    digits = spec$digits,
    big.mark = spec$big_mark,
    decimal.mark = spec$decimal_mark
  )

  if (spec$position == "prefix") {
    out <- paste0(spec$symbol, spec$sep, body)
  } else {
    out <- paste0(body, spec$sep, spec$symbol)
  }

  neg <- !is.na(x) & x < 0
  if (accounting) {
    out[neg] <- paste0("(", out[neg], ")")
  } else {
    out[neg] <- paste0("-", out[neg])
  }

  out[is.na(x)] <- na_string
  out
}

#' Format changes in monetary amounts
#'
#' Gains are prefixed with `gain_mark` and losses with `loss_mark`; no change
#' is written as `flat_label`.
#'
#' @param x A numeric vector of amounts.
#' @param code A single currency code.
#' @param gain_mark Prefix for positive changes.
#' @param loss_mark Prefix for negative changes.
#' @param flat_label String used for a change of exactly zero.
#' @return A character vector the same length as `x`.
#' @family currency formatters
#' @export
fmt_currency_delta <- function(x, code = "EUR", gain_mark = "+", loss_mark = "-",
                               flat_label = "no change") {
  out <- rep(flat_label, length(x))
  gain <- !is.na(x) & x > 0
  loss <- !is.na(x) & x < 0
  if (any(gain)) {
    out[gain] <- paste0(gain_mark, fmt_currency(x[gain], code))
  }
  if (any(loss)) {
    out[loss] <- paste0(loss_mark, fmt_currency(abs(x[loss]), code))
  }
  out[is.na(x)] <- NA_character_
  out
}

#' Convert amounts between currencies
#'
#' @param x A numeric vector of amounts.
#' @param from,to Currency codes found in `names(rates)`.
#' @param rates A named numeric vector of exchange rates per unit of `EUR`.
#' @param digits Number of decimal places to keep.
#' @return A numeric vector the same length as `x`, rounded half up.
#' @export
convert_amount <- function(x, from, to, rates, digits = 2L) {
  check_currency_args(x, from, rates)
  check_currency_args(x, to, rates)
  missing_codes <- setdiff(c(from, to), names(rates))
  if (length(missing_codes) > 0L) {
    stop(sprintf("No rate for %s.", paste(missing_codes, collapse = ", ")), call. = FALSE)
  }
  in_eur <- x / rates[[from]]
  round_amount(in_eur * rates[[to]], digits, "half_up")
}

#' Totals by group with formatted labels
#'
#' @param x A numeric vector of amounts.
#' @param group A vector of group labels the same length as `x`.
#' @param code A single currency code.
#' @param digits Number of decimal places used for the share column.
#' @return A data frame with columns `group`, `total`, `share` and `label`,
#'   ordered by decreasing `total`.
#' @export
currency_totals <- function(x, group, code = "EUR", digits = 1L) {
  if (length(x) != length(group)) {
    stop("`x` and `group` must have the same length.", call. = FALSE)
  }

  total <- tapply(x, group, sum, na.rm = TRUE)
  total <- sort(total, decreasing = TRUE)
  share <- total / sum(total)
  share_text <- formatC(100 * share, format = "f", digits = digits)
  label <- sprintf("%s (%s%%)", fmt_currency(total, code), share_text)
  data.frame(
    group = names(total),
    total = as.numeric(total),
    share = as.numeric(share),
    label = label,
    row.names = NULL
  )
}

#' Price breakdown for an invoice
#'
#' The discount is taken from the net amount; tax is charged on the
#' discounted subtotal plus shipping. Rounding is half up to cents.
#'
#' @param net A single net amount.
#' @param discount_rate Discount as a proportion between 0 and 1.
#' @param tax_rate Tax as a proportion between 0 and 1.
#' @param shipping A single non-negative shipping charge.
#' @param code A single currency code.
#' @return A data frame with columns `item`, `amount` and `label`, and one
#'   row each for `net`, `discount`, `subtotal`, `shipping`, `tax` and
#'   `gross`.
#' @export
price_breakdown <- function(net, discount_rate = 0, tax_rate = 0.19, shipping = 0,
                            code = "EUR") {
  # Validate `net`
  if (!is.numeric(net) || length(net) != 1L || is.na(net)) {
    stop("`net` must be a single number.", call. = FALSE)
  }
  # Validate `discount_rate`
  if (!is.numeric(discount_rate) || length(discount_rate) != 1L ||
      is.na(discount_rate) || discount_rate < 0 || discount_rate > 1) {
    stop("`discount_rate` must be between 0 and 1.", call. = FALSE)
  }
  # Validate `tax_rate`
  if (!is.numeric(tax_rate) || length(tax_rate) != 1L ||
      is.na(tax_rate) || tax_rate < 0 || tax_rate > 1) {
    stop("`tax_rate` must be between 0 and 1.", call. = FALSE)
  }
  # Validate `shipping`
  if (!is.numeric(shipping) || length(shipping) != 1L || is.na(shipping) || shipping < 0) {
    stop("`shipping` must be a single non-negative number.", call. = FALSE)
  }

  discount <- round_amount(net * discount_rate)
  subtotal <- net - discount
  tax <- round_amount((subtotal + shipping) * tax_rate)
  gross <- subtotal + shipping + tax

  amounts <- c(
    net = net,
    discount = -discount,
    subtotal = subtotal,
    shipping = shipping,
    tax = tax,
    gross = gross
  )
  data.frame(
    item = names(amounts),
    amount = unname(amounts),
    label = fmt_currency(unname(amounts), code),
    row.names = NULL
  )
}

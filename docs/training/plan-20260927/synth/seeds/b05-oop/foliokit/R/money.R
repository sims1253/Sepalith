#' Construct a money vector
#'
#' Low-level constructor for the `foliokit_money` class. It performs only
#' cheap type checks; use [money()] for user-facing input.
#'
#' @param x A double vector of amounts.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_money` vector.
#' @keywords internal
new_money <- function(x = double(), currency = "USD") {
  stopifnot(is.double(x))
  stopifnot(is.character(currency), length(currency) == 1L)
  vctrs::new_vctr(x, currency = currency, class = "foliokit_money")
}

#' Validate a money vector
#'
#' Checks the invariants of a `foliokit_money` vector that the constructor
#' does not: the format of the currency code and finiteness of amounts.
#'
#' @param x A `foliokit_money` vector.
#' @return `x`, invisibly, if it is valid; otherwise an error.
#' @keywords internal
validate_money <- function(x) {
  currency <- attr(x, "currency")
  if (!grepl("^[A-Z]{3}$", currency)) {
    stop("`currency` must be a three-letter upper-case code, not '", currency, "'.",
      call. = FALSE)
  }
  values <- vctrs::vec_data(x)
  if (any(is.infinite(values))) {
    stop("Money amounts must be finite.", call. = FALSE)
  }
  invisible(x)
}

#' Create a money vector
#'
#' Amounts are rounded to the number of minor units of the currency, which is
#' two for most currencies and zero for a few such as JPY.
#'
#' @param x A numeric vector of amounts.
#' @param currency A three-letter ISO currency code.
#' @return A `foliokit_money` vector.
#' @export
#' @examples
#' money(c(10, 2.5), "EUR")
money <- function(x = double(), currency = "USD") {
  currency <- toupper(currency)
  x <- round(as.double(x), minor_units(currency))
  validate_money(new_money(x, currency))
}

#' Number of minor units for a currency
#'
#' @param currency A three-letter ISO currency code.
#' @return An integer number of decimal places.
#' @keywords internal
minor_units <- function(currency) {
  zero_decimal <- c("JPY", "KRW", "ISK", "CLP")
  three_decimal <- c("KWD", "BHD", "OMR")
  if (currency %in% zero_decimal) {
    0L
  } else if (currency %in% three_decimal) {
    3L
  } else {
    2L
  }
}

#' Test whether an object is a money vector
#'
#' @param x An object to test.
#' @return `TRUE` or `FALSE`.
#' @export
is_money <- function(x) {
  inherits(x, "foliokit_money")
}

#' Get the currency of a money vector
#'
#' @param x A `foliokit_money` vector.
#' @return A single string.
#' @export
currency <- function(x) {
  if (!is_money(x)) {
    stop("`x` must be a money vector.", call. = FALSE)
  }
  attr(x, "currency")
}

#' Format a money vector
#'
#' @param x A `foliokit_money` vector.
#' @param ... Ignored.
#' @return A character vector of formatted amounts.
#' @export
format.foliokit_money <- function(x, ...) {
  cur <- attr(x, "currency")
  values <- vctrs::vec_data(x)
  out <- formatC(values, format = "f", digits = minor_units(cur), big.mark = ",")
  out[is.na(values)] <- NA_character_
  paste(cur, out)
}

#' Abbreviated type name for money vectors
#'
#' @param x A `foliokit_money` vector.
#' @param ... Ignored.
#' @return A single string used in tibble column headers.
#' @export
vec_ptype_abbr.foliokit_money <- function(x, ...) {
  tolower(attr(x, "currency"))
}

#' Ensure two money vectors share a currency
#'
#' @param x,y `foliokit_money` vectors.
#' @return The shared currency code, or an error if they differ.
#' @keywords internal
check_same_currency <- function(x, y) {
  cx <- attr(x, "currency")
  cy <- attr(y, "currency")
  if (!identical(cx, cy)) {
    stop("Can't combine amounts in ", cx, " and ", cy, ".", call. = FALSE)
  }
  cx
}

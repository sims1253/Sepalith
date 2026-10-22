#' Rebase a price index to a chosen base period
#'
#' Rescales `index` so that its value in `base` equals `to`. When `base`
#' matches several periods their average is used.
#'
#' @param index Numeric vector of price index values.
#' @param periods Vector of period labels, the same length as `index`.
#' @param base Period label (or labels) of the new base.
#' @param to Value the index should take in the base period.
#' @return A numeric vector the same length as `index`.
#' @export
rebase_index <- function(index, periods, base, to = 100) {
  if (length(index) != length(periods)) {
    stop("`index` and `periods` must have the same length.", call. = FALSE)
  }

  in_base <- periods %in% base
  if (!any(in_base)) {
    stop("`base` does not match any of `periods`.", call. = FALSE)
  }

  base_level <- mean(index[in_base], na.rm = TRUE)
  if (is.na(base_level) || base_level == 0) {
    stop("The index has no usable value in the base period.", call. = FALSE)
  }
  return(index / base_level * to)
}

#' Convert between nominal and real values
#'
#' `deflate()` expresses `values` in the prices of `base` using a price
#' index; `reflate()` is its inverse and re-applies price changes relative
#' to `base`.
#'
#' @param values Numeric vector of money amounts.
#' @param index Numeric vector of price index values.
#' @param periods Vector of period labels, the same length as `index`.
#' @param base Period label (or labels) of the new base.
#' @return A numeric vector of real or nominal values.
#' @export
deflate <- function(values, index, periods, base) {
  rebased <- rebase_index(index, periods, base, to = 1)
  return(values / rebased)
}

#' @rdname deflate
#' @export
reflate <- function(values, index, periods, base) {
  rebased <- rebase_index(index, periods, base, to = 1)
  return(values * rebased)
}

#' Period-on-period inflation rates from an index
#'
#' @param index Numeric vector of price index values.
#' @param lag Number of periods to compare across, e.g. `12` for year-on-year
#'   rates from a monthly index.
#' @return A numeric vector with `NA` for the first `lag` periods.
#' @export
inflation_rate <- function(index, lag = 1) {
  if (length(lag) != 1 || is.na(lag) || lag < 1) {
    stop("`lag` must be a single positive number.", call. = FALSE)
  }

  n <- length(index)
  out <- rep(NA_real_, n)
  if (n > lag) {
    out[(lag + 1):n] <- index[(lag + 1):n] / index[seq_len(n - lag)] - 1
  }
  return(out)
}

#' Real interest rate from a nominal rate
#'
#' Uses the exact Fisher relation unless `approx = TRUE`.
#'
#' @param nominal Nominal interest rate as a decimal.
#' @param inflation Inflation rate as a decimal.
#' @param approx Logical; use the additive approximation?
#' @return A numeric vector of real rates.
#' @family Fisher helpers
#' @export
real_rate <- function(nominal, inflation, approx = FALSE) {
  if (approx) {
    return(nominal - inflation)
  }
  return((1 + nominal) / (1 + inflation) - 1)
}

#' Nominal interest rate from a real rate
#'
#' Uses the exact Fisher relation unless `approx = TRUE`.
#'
#' @param real Real interest rate as a decimal.
#' @param inflation Inflation rate as a decimal.
#' @param approx Logical; use the additive approximation?
#' @return A numeric vector of nominal rates.
#' @family Fisher helpers
#' @export
nominal_rate <- function(real, inflation, approx = FALSE) {
  if (approx) {
    return(real + inflation)
  }
  return((1 + real) * (1 + inflation) - 1)
}

#' Convert between periodic and annual rates
#'
#' Compounds a per-period rate up to an annual rate, or splits an annual
#' rate into an equivalent per-period rate.
#'
#' @param rate Interest rate as a decimal.
#' @param per_year Number of periods per year.
#' @param direction Either `"annualise"` or `"periodise"`.
#' @param method Either `"compound"` or `"simple"`.
#' @return A numeric vector of converted rates.
#' @export
convert_rate <- function(rate, per_year = 12, direction = c("annualise", "periodise"),
                         method = c("compound", "simple")) {
  direction <- match.arg(direction)
  method <- match.arg(method)
  check_rate(rate)

  # one switch over both choices keeps the four formulas side by side
  out <- switch(
    paste(direction, method),
    "annualise compound" = (1 + rate)^per_year - 1,
    "annualise simple" = rate * per_year,
    "periodise compound" = (1 + rate)^(1 / per_year) - 1,
    "periodise simple" = rate / per_year
  )
  return(out)
}

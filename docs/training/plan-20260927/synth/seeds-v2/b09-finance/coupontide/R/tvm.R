#' Present and future value of a single amount
#'
#' Discounts a cash amount received after `n` periods back to today, or
#' compounds an amount held today forward by `n` periods, using a constant
#' per-period rate.
#'
#' @param amount Numeric vector of cash amounts.
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param n Number of periods; may be fractional.
#' @return A numeric vector of values, recycled to the longest input.
#' @export
present_value <- function(amount, rate, n) {
  check_rate(rate)
  if (any(n < 0, na.rm = TRUE)) {
    stop("`n` must be non-negative.", call. = FALSE)
  }

  return(amount / (1 + rate)^n)
}

#' @rdname present_value
#' @export
future_value <- function(amount, rate, n) {
  check_rate(rate)
  if (any(n < 0, na.rm = TRUE)) {
    stop("`n` must be non-negative.", call. = FALSE)
  }

  return(amount * (1 + rate)^n)
}

#' Present value of a level annuity
#'
#' Values a stream of `n` equal payments. Payments fall at the end of each
#' period unless `due = TRUE`, in which case they fall at the start.
#'
#' @param payment Payment made each period.
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param n Number of periods; may be fractional.
#' @param due Logical; are payments made at the start of each period?
#' @return A numeric vector of annuity present values.
#' @family annuity helpers
#' @export
annuity_pv <- function(payment, rate, n, due = FALSE) {
  check_rate(rate)
  # a zero rate makes the closed form 0/0, so fall back to a plain count
  factor <- ifelse(rate == 0, n, (1 - (1 + rate)^(-n)) / rate)
  if (due) {
    factor <- factor * (1 + rate)
  }
  return(payment * factor)
}

#' Future value of a level annuity
#'
#' Accumulates a stream of `n` equal payments to the end of the final
#' period. Payments fall at the end of each period unless `due = TRUE`.
#'
#' @param payment Payment made each period.
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param n Number of periods; may be fractional.
#' @param due Logical; are payments made at the start of each period?
#' @return A numeric vector of annuity future values.
#' @family annuity helpers
#' @export
annuity_fv <- function(payment, rate, n, due = FALSE) {
  check_rate(rate)
  factor <- ifelse(rate == 0, n, ((1 + rate)^n - 1) / rate)
  if (due) {
    factor <- factor * (1 + rate)
  }
  return(payment * factor)
}

#' Level payment that repays a principal
#'
#' Solves the annuity equation for the payment that amortises `principal`
#' over `n` periods, optionally leaving a `balloon` amount outstanding.
#'
#' @param principal Amount borrowed.
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param n Number of periods; may be fractional.
#' @param balloon Amount still owed after the final payment.
#' @return A numeric vector of per-period payments.
#' @export
annuity_payment <- function(principal, rate, n, balloon = 0) {
  check_rate(rate)
  if (any(n <= 0, na.rm = TRUE)) {
    stop("`n` must be positive.", call. = FALSE)
  }

  # only the part of the principal not covered by the balloon is amortised
  financed <- principal - present_value(balloon, rate, n)
  payment <- ifelse(rate == 0, financed / n, financed * rate / (1 - (1 + rate)^(-n)))
  return(payment)
}

#' Net present value of an irregular cash flow
#'
#' Discounts each element of `cash_flows` by its position in the vector.
#' The first element is treated as time zero unless `start = 1`.
#'
#' @param cash_flows Numeric vector of cash flows in period order.
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param start Period index of the first cash flow, usually `0` or `1`.
#' @return A single numeric net present value.
#' @seealso [internal_rate()]
#' @export
net_present_value <- function(cash_flows, rate, start = 0) {
  check_rate(rate)
  if (length(cash_flows) == 0) {
    return(0)
  }

  periods <- seq_along(cash_flows) - 1 + start
  return(sum(cash_flows / (1 + rate)^periods))
}

#' Internal rate of return
#'
#' Finds the rate at which [net_present_value()] of `cash_flows` is zero by
#' bisection between `lower` and `upper`.
#'
#' @param cash_flows Numeric vector of cash flows in period order.
#' @param lower,upper Bracketing rates for the search.
#' @param tol Convergence tolerance on the rate.
#' @param max_iter Maximum number of bisection steps.
#' @return A single numeric rate, or `NA` when no sign change is bracketed
#'   or a cash flow is missing.
#' @seealso [net_present_value()]
#' @export
internal_rate <- function(cash_flows, lower = -0.99, upper = 1, tol = 1e-8,
                          max_iter = 200) {
  if (anyNA(cash_flows)) {
    warning("`cash_flows` contains missing values.", call. = FALSE)
    return(NA_real_)
  }
  if (!any(cash_flows < 0) || !any(cash_flows > 0)) {
    warning("Cash flows need at least one sign change.", call. = FALSE)
    return(NA_real_)
  }

  f_lower <- net_present_value(cash_flows, lower)
  f_upper <- net_present_value(cash_flows, upper)
  if (f_lower == 0) {
    return(lower)
  }
  if (f_upper == 0) {
    return(upper)
  }
  if (sign(f_lower) == sign(f_upper)) {
    return(NA_real_)
  }

  # plain bisection: slow but cannot overshoot the bracket
  for (i in seq_len(max_iter)) {
    mid <- (lower + upper) / 2
    f_mid <- net_present_value(cash_flows, mid)
    if (abs(f_mid) < tol || (upper - lower) / 2 < tol) {
      return(mid)
    }
    if (sign(f_mid) == sign(f_lower)) {
      lower <- mid
      f_lower <- f_mid
    } else {
      upper <- mid
    }
  }

  warning("`internal_rate()` did not converge.", call. = FALSE)
  return((lower + upper) / 2)
}

#' Validate an interest rate argument
#'
#' @param rate Per-period interest rate as a decimal (e.g. `0.05`).
#' @param arg Name of the argument, used in error messages.
#' @noRd
check_rate <- function(rate, arg = "rate") {
  if (!is.numeric(rate)) {
    stop(sprintf("`%s` must be numeric.", arg), call. = FALSE)
  }
  if (any(rate <= -1, na.rm = TRUE)) {
    stop(sprintf("`%s` must be greater than -1.", arg), call. = FALSE)
  }
  return(invisible(rate))
}

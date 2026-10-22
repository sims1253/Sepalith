#' Build a loan amortisation schedule
#'
#' Computes the level payment for a fixed-rate loan and splits each payment
#' into interest and principal, period by period.
#'
#' @param principal Amount borrowed.
#' @param annual_rate Nominal annual interest rate as a decimal.
#' @param years Loan term in years.
#' @param per_year Number of payments per year.
#' @param extra Additional principal paid each period; recycled to the term.
#' @return An `amort_schedule` data frame with one row per payment.
#' @family amortisation helpers
#' @export
amortisation_schedule <- function(principal, annual_rate, years, per_year = 12,
                                  extra = 0) {
  check_rate(annual_rate, "annual_rate")
  n <- round(years * per_year)
  if (n < 1) {
    stop("The loan term must cover at least one payment.", call. = FALSE)
  }
  rate <- annual_rate / per_year
  payment <- annuity_payment(principal, rate, n)
  extra <- rep_len(extra, n)

  interest <- numeric(n)
  repaid <- numeric(n)
  balance <- numeric(n)
  opening <- principal

  for (i in seq_len(n)) {
    interest[i] <- opening * rate
    # never repay more than is still owed, even with extra payments
    repaid[i] <- min(payment - interest[i] + extra[i], opening)
    balance[i] <- opening - repaid[i]
    opening <- balance[i]
    if (balance[i] <= 1e-8) {
      n <- i
      break
    }
  }

  idx <- seq_len(n)
  schedule <- data.frame(
    period = idx,
    payment = interest[idx] + repaid[idx],
    interest = interest[idx],
    principal = repaid[idx],
    balance = pmax(balance[idx], 0)
  )
  return(new_amort_schedule(schedule, principal, annual_rate, per_year))
}

#' Construct an amortisation schedule object
#'
#' @param schedule A data frame with `period`, `payment`, `interest`,
#'   `principal` and `balance` columns.
#' @param principal Amount borrowed.
#' @param annual_rate Nominal annual interest rate as a decimal.
#' @param per_year Number of payments per year.
#' @return `schedule` with class `amort_schedule` and loan attributes.
#' @keywords internal
new_amort_schedule <- function(schedule, principal, annual_rate, per_year) {
  stopifnot(is.data.frame(schedule))
  attr(schedule, "principal") <- principal
  attr(schedule, "annual_rate") <- annual_rate
  attr(schedule, "per_year") <- per_year
  class(schedule) <- c("amort_schedule", class(schedule))
  return(schedule)
}

#' Outstanding balance after a given number of payments
#'
#' Uses the closed form rather than building the full schedule, which is
#' convenient for quick payoff quotes.
#'
#' @param principal Amount borrowed.
#' @param annual_rate Nominal annual interest rate as a decimal.
#' @param years Loan term in years.
#' @param paid Number of payments already made.
#' @param per_year Number of payments per year.
#' @return A numeric vector of remaining balances.
#' @export
remaining_balance <- function(principal, annual_rate, years, paid, per_year = 12) {
  rate <- annual_rate / per_year
  # round like amortisation_schedule() so both agree on the payment
  n <- round(years * per_year)
  payment <- annuity_payment(principal, rate, n)

  # balance = principal grown forward minus the payments grown forward
  balance <- future_value(principal, rate, paid) - annuity_fv(payment, rate, paid)
  return(pmax(balance, 0))
}

#' Total interest paid over a schedule
#'
#' @param schedule An `amort_schedule` object.
#' @return A single number.
#' @export
total_interest <- function(schedule) {
  stopifnot(inherits(schedule, "amort_schedule"))
  return(sum(schedule$interest))
}

#' Aggregate a schedule to calendar years
#'
#' Sums payments, interest and principal within each loan year and reports
#' the closing balance at the end of the year.
#'
#' @param schedule An `amort_schedule` object.
#' @return A data frame with one row per loan year.
#' @export
annual_summary <- function(schedule) {
  stopifnot(inherits(schedule, "amort_schedule"))
  per_year <- attr(schedule, "per_year")
  year <- (schedule$period - 1) %/% per_year + 1

  sums <- rowsum(schedule[c("payment", "interest", "principal")], year)
  closing <- tapply(schedule$balance, year, function(b) b[length(b)])

  out <- data.frame(
    year = as.integer(rownames(sums)),
    sums,
    balance = as.numeric(closing),
    row.names = NULL
  )
  return(out)
}

#' Format an amortisation schedule
#'
#' @param x An `amort_schedule` object.
#' @param digits Number of decimal places for money columns.
#' @param ... Ignored.
#' @return A character data frame suitable for printing.
#' @export
format.amort_schedule <- function(x, digits = 2, ...) {
  money <- c("payment", "interest", "principal", "balance")
  out <- as.data.frame(unclass(x))[c("period", money)]
  for (col in money) {
    out[[col]] <- formatC(out[[col]], format = "f", digits = digits, big.mark = ",")
  }
  return(out)
}

#' Print an amortisation schedule
#'
#' @param x An `amort_schedule` object.
#' @param n Number of rows to show from the start of the schedule.
#' @param ... Passed to [format.amort_schedule()].
#' @return `x`, invisibly.
#' @export
print.amort_schedule <- function(x, n = 12, ...) {
  cat(sprintf(
    "<amort_schedule> %s payments, principal %s at %.3f%% p.a.\n",
    nrow(x), format(attr(x, "principal"), big.mark = ",", scientific = FALSE),
    100 * attr(x, "annual_rate")
  ))
  print(utils::head(format(x, ...), n), row.names = FALSE)

  if (nrow(x) > n) {
    cat(sprintf("# ... with %d more payments\n", nrow(x) - n))
  }
  return(invisible(x))
}

#' Summarise an amortisation schedule
#'
#' @param object An `amort_schedule` object.
#' @param ... Ignored.
#' @return A named numeric vector of loan totals.
#' @export
summary.amort_schedule <- function(object, ...) {
  out <- c(
    principal = attr(object, "principal"),
    payments = nrow(object),
    total_paid = sum(object$payment),
    total_interest = total_interest(object),
    interest_share = total_interest(object) / sum(object$payment)
  )
  return(out)
}

#' Compare refinancing an existing loan
#'
#' Prices the remaining balance of the current loan under a new rate and
#' term, including up-front costs, and reports the payment change, interest
#' saved and the number of months needed to recoup the costs.
#'
#' @param principal Amount borrowed.
#' @param annual_rate Nominal annual interest rate as a decimal.
#' @param years Loan term in years.
#' @param paid Number of payments already made.
#' @param new_rate Nominal annual rate of the replacement loan.
#' @param new_years Term of the replacement loan in years.
#' @param costs Up-front refinancing costs, added to the new loan if
#'   `finance_costs = TRUE`.
#' @param finance_costs Logical; roll `costs` into the new principal?
#' @param per_year Number of payments per year.
#' @return A named list of comparison figures.
#' @family amortisation helpers
#' @export
refinance_comparison <- function(principal, annual_rate, years, paid, new_rate,
                                 new_years, costs = 0, finance_costs = FALSE,
                                 per_year = 12) {
  n_total <- round(years * per_year)
  if (paid >= n_total) {
    stop("The existing loan is already repaid.", call. = FALSE)
  }

  # what is left on the current loan if nothing changes
  balance <- remaining_balance(principal, annual_rate, years, paid, per_year)
  old_payment <- annuity_payment(principal, annual_rate / per_year, n_total)
  old_remaining <- n_total - paid
  old_interest <- old_payment * old_remaining - balance

  # the replacement loan starts from today's balance
  new_principal <- if (finance_costs) balance + costs else balance
  new_n <- round(new_years * per_year)
  new_payment <- annuity_payment(new_principal, new_rate / per_year, new_n)
  new_interest <- new_payment * new_n - new_principal

  monthly_saving <- old_payment - new_payment
  upfront <- if (finance_costs) 0 else costs
  breakeven <- if (monthly_saving > 0) {
    ceiling(upfront / monthly_saving)
  } else {
    NA_integer_
  }

  out <- list(
    balance = balance,
    old_payment = old_payment,
    new_payment = new_payment,
    payment_change = new_payment - old_payment,
    old_total_cost = old_payment * old_remaining,
    new_total_cost = new_payment * new_n + upfront,
    interest_saved = old_interest - new_interest,
    breakeven_payments = breakeven
  )
  return(out)
}

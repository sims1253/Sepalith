#' Growth factor of one unit invested at a nominal rate
#'
#' Periods per year come from the table `perYear`: `annual` 1,
#' `semiannual` 2, `quarterly` 4, `monthly` 12. `simple` interest grows
#' linearly and `continuous` compounding uses the exponential.
#'
#' @param rate Nominal annual interest rate as a decimal.
#' @param time Time in years.
#' @param compounding One of `"simple"`, `"annual"`, `"semiannual"`,
#'   `"quarterly"`, `"monthly"` or `"continuous"`.
#' @return A numeric vector of growth factors.
#' @family interest
#' @export
compoundFactor = function(rate, time,
                          compounding = c("simple", "annual", "semiannual", "quarterly",
                                          "monthly", "continuous")) {
  compounding = match.arg(compounding)
  perYear = c(annual = 1, semiannual = 2, quarterly = 4, monthly = 12)

  switch(compounding,
    simple = {
      1 + rate * time
    },
    annual = {
      m = perYear[["annual"]]
      (1 + rate / m)^(m * time)
    },
    semiannual = {
      m = perYear[["semiannual"]]
      (1 + rate / m)^(m * time)
    },
    quarterly = {
      m = perYear[["quarterly"]]
      (1 + rate / m)^(m * time)
    },
    monthly = {
      m = perYear[["monthly"]]
      (1 + rate / m)^(m * time)
    },
    continuous = {
      exp(rate * time)
    }
  )
}

#' Discount factor of one unit due in the future
#'
#' The mirror image of [compoundFactor()]: the present value of one unit
#' paid at `time`, with periods per year taken from the same `perYear`
#' table.
#'
#' @param rate Nominal annual interest rate as a decimal.
#' @param time Time in years.
#' @param compounding One of `"simple"`, `"annual"`, `"semiannual"`,
#'   `"quarterly"`, `"monthly"` or `"continuous"`.
#' @return A numeric vector of discount factors.
#' @family interest
#' @export
discountFactor = function(rate, time,
                          compounding = c("simple", "annual", "semiannual", "quarterly",
                                          "monthly", "continuous")) {
  compounding = match.arg(compounding)
  perYear = c(annual = 1, semiannual = 2, quarterly = 4, monthly = 12)

  switch(compounding,
    simple = {
      1 / (1 + rate * time)
    },
    annual = {
      m = perYear[["annual"]]
      (1 + rate / m)^(-m * time)
    },
    semiannual = {
      m = perYear[["semiannual"]]
      (1 + rate / m)^(-m * time)
    },
    quarterly = {
      m = perYear[["quarterly"]]
      (1 + rate / m)^(-m * time)
    },
    monthly = {
      m = perYear[["monthly"]]
      (1 + rate / m)^(-m * time)
    },
    continuous = {
      exp(-rate * time)
    }
  )
}

#' Convert a nominal rate between compounding conventions
#'
#' The rate is first turned into the equivalent effective annual rate and
#' then expressed under the target convention. Both conventions take their
#' periods per year from `perYear`, as in [compoundFactor()].
#'
#' @param rate Nominal annual interest rate as a decimal.
#' @param from,to Compounding conventions: `"annual"`, `"semiannual"`,
#'   `"quarterly"`, `"monthly"` or `"continuous"`.
#' @return The equivalent nominal rate under `to`.
#' @family interest
#' @export
convertRate = function(rate, from, to) {
  choices = c("annual", "semiannual", "quarterly", "monthly", "continuous")
  from = match.arg(from, choices)
  to = match.arg(to, choices)
  if (!is.numeric(rate) || anyNA(rate)) {
    stop("`rate` must be a numeric vector without missing values.", call. = FALSE)
  }
  perYear = c(annual = 1, semiannual = 2, quarterly = 4, monthly = 12)

  # Step 1: nominal under `from` to effective annual
  if (from == "continuous") {
    effective = exp(rate) - 1
  } else {
    m = perYear[[from]]
    effective = (1 + rate / m)^m - 1
  }
  # Step 2: effective annual to nominal under `to`
  if (to == "continuous") {
    log(1 + effective)
  } else {
    m = perYear[[to]]
    m * ((1 + effective)^(1 / m) - 1)
  }
}

#' Value of a level annuity
#'
#' The present value discounts each payment back to time zero; the future
#' value mirrors it by growing each payment forward to the last period.
#' Payments in advance (`timing = "begin"`) are worth one extra period of
#' interest under both.
#'
#' @param payment Level payment per period.
#' @param rate Interest rate per payment period as a decimal.
#' @param n Number of payments.
#' @param value `"present"` or `"future"`.
#' @param timing `"end"` for payments in arrears, `"begin"` for payments in
#'   advance.
#' @return A single number.
#' @family interest
#' @export
annuityValue = function(payment, rate, n, value = c("present", "future"),
                        timing = c("end", "begin")) {
  value = match.arg(value)
  timing = match.arg(timing)
  if (n < 1 || n != round(n)) stop("`n` must be a positive whole number.", call. = FALSE)

  if (value == "present") {
    # Discount every payment to time zero
    factor = if (rate == 0) n else (1 - (1 + rate)^-n) / rate
    total = payment * factor
  } else {
    # Grow every payment to period n
    factor = if (rate == 0) n else ((1 + rate)^n - 1) / rate
    total = payment * factor
  }
  if (timing == "begin") {
    total = total * (1 + rate)
  }
  total
}

#' Investment measures of a cash flow stream
#'
#' @param cashflows Cash flows per period, the first at time zero.
#' @param rate Discount rate per period as a decimal.
#' @return A list with elements `npv`, `inflows` (present value of the
#'   positive flows), `outflows` (present value of the negative flows, as a
#'   positive number), `ratio` (`inflows / outflows`), `payback` and
#'   `discountedPayback` (first period whose cumulative flow is
#'   non-negative, `NA` if never).
#' @family interest
#' @export
cashFlowMeasures = function(cashflows, rate) {
  times = seq_along(cashflows) - 1
  discounted = cashflows / (1 + rate)^times

  # Present value of each side
  inflows = sum(discounted[discounted > 0])
  outflows = -sum(discounted[discounted < 0])
  # Undiscounted payback
  reached = which(cumsum(cashflows) >= 0 & times > 0)
  payback = if (length(reached)) times[reached[1L]] else NA_real_
  # Discounted payback
  reached = which(cumsum(discounted) >= 0 & times > 0)
  discountedPayback = if (length(reached)) times[reached[1L]] else NA_real_

  list(
    npv = inflows - outflows,
    inflows = inflows,
    outflows = outflows,
    ratio = if (outflows > 0) inflows / outflows else NA_real_,
    payback = payback,
    discountedPayback = discountedPayback
  )
}

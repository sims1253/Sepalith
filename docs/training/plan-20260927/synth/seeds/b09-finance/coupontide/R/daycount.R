#' Year fraction under the Actual/360 convention
#'
#' @param start,end Date vectors marking the start and end of each accrual
#'   period.
#' @return A numeric vector of year fractions.
#' @family day count conventions
#' @export
yearfrac_act360 <- function(start, end) {
  days <- as.numeric(as.Date(end) - as.Date(start))
  return(days / 360)
}

#' Year fraction under the Actual/365 Fixed convention
#'
#' @param start,end Date vectors marking the start and end of each accrual
#'   period.
#' @return A numeric vector of year fractions.
#' @family day count conventions
#' @export
yearfrac_act365 <- function(start, end) {
  days <- as.numeric(as.Date(end) - as.Date(start))
  return(days / 365)
}

#' Year fraction under the 30/360 US convention
#'
#' Treats every month as 30 days, adjusting day 31 (and the end of
#' February for the start date) before counting.
#'
#' @param start,end Date vectors marking the start and end of each accrual
#'   period.
#' @return A numeric vector of year fractions.
#' @family day count conventions
#' @export
yearfrac_30360 <- function(start, end) {
  s <- as.POSIXlt(as.Date(start))
  e <- as.POSIXlt(as.Date(end))
  d1 <- s$mday
  d2 <- e$mday
  # last day of February counts as the 30th
  feb_end <- s$mon == 1 & as.POSIXlt(as.Date(start) + 1)$mon == 2
  d1[feb_end] <- 30
  d1[d1 == 31] <- 30
  d2[d2 == 31 & d1 == 30] <- 30
  days <- 360 * (e$year - s$year) + 30 * (e$mon - s$mon) + (d2 - d1)
  return(days / 360)
}

#' Year fraction under a named day count convention
#'
#' @param start,end Date vectors marking the start and end of each accrual
#'   period.
#' @param convention Day count convention to apply.
#' @return A numeric vector of year fractions.
#' @family day count conventions
#' @export
year_fraction <- function(start, end, convention = c("act365", "act360", "30360")) {
  convention <- match.arg(convention)
  frac <- switch(convention,
    act365 = yearfrac_act365(start, end),
    act360 = yearfrac_act360(start, end),
    "30360" = yearfrac_30360(start, end)
  )
  return(frac)
}

#' Shift dates by whole months
#'
#' Adds `months` to each date, clamping to the last day of the target month
#' so that 31 January plus one month gives the end of February.
#'
#' @param dates A Date vector.
#' @param months Integer number of months to add; may be negative.
#' @return A Date vector the same length as `dates`.
#' @export
add_months <- function(dates, months) {
  lt <- as.POSIXlt(as.Date(dates))
  total <- lt$year * 12 + lt$mon + months
  year <- total %/% 12 + 1900
  month <- total %% 12 + 1
  first <- as.Date(sprintf("%04d-%02d-01", year, month))
  last_day <- as.POSIXlt(add_one_month(first) - 1)$mday
  day <- pmin(lt$mday, last_day)
  return(first + day - 1)
}

#' First day of the following month
#'
#' @param first A Date vector of first-of-month dates.
#' @return A Date vector.
#' @noRd
add_one_month <- function(first) {
  lt <- as.POSIXlt(first)
  lt$mon <- lt$mon + 1
  return(as.Date(lt))
}

#' Regular coupon or payment dates
#'
#' Generates payment dates backwards from `maturity` so that any short stub
#' period falls at the start.
#'
#' @param issue Issue date of the instrument.
#' @param maturity Final payment date.
#' @param per_year Number of payments per year; must divide 12.
#' @return A sorted Date vector of payment dates after `issue`.
#' @export
payment_dates <- function(issue, maturity, per_year = 2) {
  issue <- as.Date(issue)
  maturity <- as.Date(maturity)
  if (maturity <= issue) {
    stop("`maturity` must be after `issue`.", call. = FALSE)
  }
  if (12 %% per_year != 0) {
    stop("`per_year` must divide 12.", call. = FALSE)
  }
  step <- 12 %/% per_year
  max_n <- ceiling(as.numeric(maturity - issue) / 365 * per_year) + 1
  dates <- add_months(maturity, -step * seq(0, max_n))
  return(sort(dates[dates > issue]))
}

#' Interest accrued since the last payment date
#'
#' @param face Face value of the instrument.
#' @param coupon_rate Annual coupon rate as a decimal.
#' @param last_paid Date of the most recent payment.
#' @param settle Settlement date.
#' @param convention Day count convention to apply.
#' @return A numeric vector of accrued interest amounts.
#' @export
accrued_interest <- function(face, coupon_rate, last_paid, settle,
                             convention = c("act365", "act360", "30360")) {
  convention <- match.arg(convention)
  if (any(as.Date(settle) < as.Date(last_paid))) {
    stop("`settle` must not precede `last_paid`.", call. = FALSE)
  }
  frac <- year_fraction(last_paid, settle, convention)
  return(face * coupon_rate * frac)
}

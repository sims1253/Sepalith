#' Outstanding balance of an annuity loan
#'
#' @param principal Amount borrowed.
#' @param rate Interest rate per period as a decimal.
#' @param n Total number of payments.
#' @param k Number of payments already made.
#' @return The balance after payment `k`.
#' @keywords internal
remainingBalance = function(principal, rate, n, k) {
  if (rate == 0) {
    return(principal * (n - k) / n)
  }
  growth = (1 + rate)^c(n, k)
  principal * (growth[1] - growth[2]) / (growth[1] - 1)
}

#' Loan amortization schedule
#'
#' @details
#' The principal repaid in each period depends on `type`:
#' * `"annuity"`: level total payments;
#' * `"linear"`: level principal repayments;
#' * `"bullet"`: interest only, principal repaid in the last period;
#' * `"balloon"`: annuity payments sized on the full principal, with
#'   `balloon` left over and repaid in the last period.
#'
#' @param principal Amount borrowed.
#' @param rate Interest rate per period as a decimal.
#' @param n Number of periods.
#' @param type One of `"annuity"`, `"linear"`, `"bullet"` or `"balloon"`.
#' @param balloon Residual amount repaid at the end for `type = "balloon"`.
#' @return A data frame with columns `period`, `opening`, `interest`,
#'   `principal`, `payment` and `closing`.
#' @export
amortizationSchedule = function(principal, rate, n,
                                type = c("annuity", "linear", "bullet", "balloon"),
                                balloon = 0) {
  type = match.arg(type)
  opening = numeric(n)
  interest = numeric(n)
  repaid = numeric(n)
  balance = principal

  for (i in seq_len(n)) {
    opening[i] = balance
    interest[i] = balance * rate
    repaid[i] = switch(type,
      annuity = {
        payment = principal * rate / (1 - (1 + rate)^-n)
        if (rate == 0) principal / n else payment - interest[i]
      },
      linear = {
        principal / n
      },
      bullet = {
        if (i == n) balance else 0
      },
      balloon = {
        amortizing = principal - balloon / (1 + rate)^n
        payment = amortizing * rate / (1 - (1 + rate)^-n)
        if (i == n) balance else payment - interest[i]
      }
    )
    balance = balance - repaid[i]
  }

  data.frame(
    period = seq_len(n),
    opening = opening,
    interest = interest,
    principal = repaid,
    payment = interest + repaid,
    closing = opening - repaid
  )
}

#' Totals of an amortization schedule
#'
#' @param schedule A schedule as returned by [amortizationSchedule()].
#' @param periodsPerYear Number of schedule periods per year, used to group
#'   the totals by year.
#' @return A list with elements `periods`, `finalBalance`, `overall` (a
#'   named vector of the `interest`, `principal` and `payment` totals) and
#'   `byYear` (a data frame with columns `year`, `interest`, `principal`
#'   and `payment`).
#' @export
scheduleTotals = function(schedule, periodsPerYear = 12L) {
  year = (schedule$period - 1L) %/% periodsPerYear + 1L
  byYear = data.frame(year = sort(unique(year)))

  # Column `interest`
  sums = tapply(schedule$interest, year, sum)
  byYear$interest = as.numeric(sums[as.character(byYear$year)])
  # Column `principal`
  sums = tapply(schedule$principal, year, sum)
  byYear$principal = as.numeric(sums[as.character(byYear$year)])
  # Column `payment`
  sums = tapply(schedule$payment, year, sum)
  byYear$payment = as.numeric(sums[as.character(byYear$year)])

  overall = c(
    interest = sum(schedule$interest),
    principal = sum(schedule$principal),
    payment = sum(schedule$payment)
  )
  list(
    periods = nrow(schedule),
    finalBalance = schedule$closing[nrow(schedule)],
    overall = overall,
    byYear = byYear
  )
}

#' Effective interest amortization of a bond premium or discount
#'
#' A bond bought at a premium has interest expense below the coupon, and
#' the difference reduces the carrying amount; a bond bought at a discount
#' mirrors this, with expense above the coupon increasing the carrying
#' amount. Both converge to `face` at maturity.
#'
#' @param carrying Initial carrying amount (price paid).
#' @param face Face value of the bond.
#' @param couponRate Coupon rate per period as a decimal.
#' @param effectiveRate Market yield per period as a decimal.
#' @param n Number of coupon periods.
#' @return A data frame with columns `period`, `opening`, `coupon`,
#'   `expense`, `amortized` and `closing`.
#' @export
bondAmortization = function(carrying, face, couponRate, effectiveRate, n) {
  coupon = face * couponRate
  opening = numeric(n)
  expense = numeric(n)
  amortized = numeric(n)
  value = carrying

  for (i in seq_len(n)) {
    opening[i] = value
    expense[i] = value * effectiveRate
    if (carrying >= face) {
      # Premium: coupon exceeds expense, carrying amount falls
      amortized[i] = coupon - expense[i]
      value = value - amortized[i]
    } else {
      # Discount: expense exceeds coupon, carrying amount rises
      amortized[i] = expense[i] - coupon
      value = value + amortized[i]
    }
  }

  data.frame(
    period = seq_len(n),
    opening = opening,
    coupon = coupon,
    expense = expense,
    amortized = amortized,
    closing = round(c(opening[-1L], value), 8)
  )
}

#' Render an amortization schedule as text
#'
#' @param schedule A schedule as returned by [amortizationSchedule()].
#' @param format One of `"csv"`, `"tsv"`, `"markdown"` or `"fixed"`.
#' @param digits Number of decimal places for money columns.
#' @return A character vector with one element per output line, header
#'   first.
#' @export
renderSchedule = function(schedule, format = c("csv", "tsv", "markdown", "fixed"),
                          digits = 2L) {
  format = match.arg(format)
  money = vapply(schedule, is.double, logical(1))
  cells = schedule
  cells[money] = lapply(schedule[money], formatC, format = "f", digits = digits)
  cells = as.data.frame(lapply(cells, as.character), stringsAsFactors = FALSE)

  switch(format,
    csv = {
      header = paste(names(cells), collapse = ",")
      body = do.call(paste, c(cells, sep = ","))
      c(header, body)
    },
    tsv = {
      header = paste(names(cells), collapse = "\t")
      body = do.call(paste, c(cells, sep = "\t"))
      c(header, body)
    },
    markdown = {
      header = paste0("| ", paste(names(cells), collapse = " | "), " |")
      rule = paste0("|", paste(rep("---:", ncol(cells)), collapse = "|"), "|")
      body = paste0("| ", do.call(paste, c(cells, sep = " | ")), " |")
      c(header, rule, body)
    },
    fixed = {
      widths = pmax(nchar(names(cells)), vapply(cells, function(x) max(nchar(x)), 1))
      padded = Map(formatC, cells, width = widths)
      header = paste(unlist(Map(formatC, names(cells), width = widths)), collapse = " ")
      body = do.call(paste, c(unname(padded), sep = " "))
      c(header, body)
    }
  )
}

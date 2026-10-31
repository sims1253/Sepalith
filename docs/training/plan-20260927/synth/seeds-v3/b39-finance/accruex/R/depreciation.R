#' Per-method depreciation schedules
#'
#' Each helper returns the full schedule of an asset under one method. The
#' depreciable base is `cost - salvage`; the final period absorbs rounding
#' so that the closing book value equals `salvage` exactly.
#'
#' @param cost Acquisition cost of the asset.
#' @param salvage Expected residual value at the end of the life.
#' @param life Useful life in whole periods.
#' @return A data frame with columns `period`, `opening`, `charge`,
#'   `accumulated` and `closing`, one row per period.
#' @name depreciationMethods
#' @keywords internal
NULL

#' @rdname depreciationMethods
straightLineSchedule = function(cost, salvage, life) {
  base = cost - salvage
  charge = rep(round(base / life, 2), life)

  # Push rounding into the last period
  charge[life] = base - sum(charge[-life])
  accumulated = cumsum(charge)
  data.frame(
    period = seq_len(life),
    opening = cost - c(0, accumulated[-life]),
    charge = charge,
    accumulated = accumulated,
    closing = cost - accumulated
  )
}

#' @rdname depreciationMethods
#' @param factor Multiple of the straight-line rate, e.g. `2` for double
#'   declining balance.
decliningSchedule = function(cost, salvage, life, factor) {
  base = cost - salvage
  charge = numeric(life)
  book = cost
  for (i in seq_len(life)) {
    # Switch to straight line once it gives the larger charge
    straight = (book - salvage) / (life - i + 1)
    charge[i] = round(min(max(book * factor / life, straight), book - salvage), 2)
    book = book - charge[i]
  }

  # Push rounding into the last period
  charge[life] = base - sum(charge[-life])
  accumulated = cumsum(charge)
  data.frame(
    period = seq_len(life),
    opening = cost - c(0, accumulated[-life]),
    charge = charge,
    accumulated = accumulated,
    closing = cost - accumulated
  )
}

#' @rdname depreciationMethods
sumOfYearsSchedule = function(cost, salvage, life) {
  base = cost - salvage
  digits = life * (life + 1) / 2
  charge = round(base * rev(seq_len(life)) / digits, 2)

  # Push rounding into the last period
  charge[life] = base - sum(charge[-life])
  accumulated = cumsum(charge)
  data.frame(
    period = seq_len(life),
    opening = cost - c(0, accumulated[-life]),
    charge = charge,
    accumulated = accumulated,
    closing = cost - accumulated
  )
}

#' @rdname depreciationMethods
#' @param units Units produced in each period; its length sets the life.
#' @param totalUnits Total units expected over the life.
unitsSchedule = function(cost, salvage, units, totalUnits) {
  base = cost - salvage
  life = length(units)
  charge = round(diff(c(0, pmin(cumsum(units) / totalUnits, 1))) * base, 2)

  # Push rounding into the last period only when the base is used up
  if (sum(units) >= totalUnits) charge[life] = base - sum(charge[-life])
  accumulated = cumsum(charge)
  data.frame(
    period = seq_len(life),
    opening = cost - c(0, accumulated[-life]),
    charge = charge,
    accumulated = accumulated,
    closing = cost - accumulated
  )
}

#' Depreciation schedule by method
#'
#' The declining balance methods take their multiple from the factor table
#' `factors`: `decliningOneHalf` 1.5, `doubleDeclining` 2 and
#' `tripleDeclining` 3.
#'
#' @param cost Acquisition cost of the asset.
#' @param salvage Expected residual value at the end of the life.
#' @param life Useful life in whole periods; for `method = "units"` the
#'   total expected units.
#' @param method One of `"straightLine"`, `"decliningOneHalf"`,
#'   `"doubleDeclining"`, `"tripleDeclining"`, `"sumOfYears"` or `"units"`.
#' @param units For `method = "units"`, the units produced in each period.
#' @return A schedule data frame with columns `period`, `opening`,
#'   `charge`, `accumulated` and `closing`, plus attribute `method`.
#' @export
depreciationSchedule = function(cost, salvage, life,
                                method = c("straightLine", "decliningOneHalf",
                                           "doubleDeclining", "tripleDeclining",
                                           "sumOfYears", "units"),
                                units = NULL) {
  method = match.arg(method)
  if (!is.numeric(cost) || cost <= 0 || salvage < 0 || salvage > cost) {
    stop("`cost` must be positive and `salvage` between 0 and `cost`.", call. = FALSE)
  }
  factors = c(decliningOneHalf = 1.5, doubleDeclining = 2, tripleDeclining = 3)

  out = switch(method,
    straightLine = {
      straightLineSchedule(cost, salvage, life)
    },
    decliningOneHalf = {
      decliningSchedule(cost, salvage, life, factors[["decliningOneHalf"]])
    },
    doubleDeclining = {
      decliningSchedule(cost, salvage, life, factors[["doubleDeclining"]])
    },
    tripleDeclining = {
      decliningSchedule(cost, salvage, life, factors[["tripleDeclining"]])
    },
    sumOfYears = {
      sumOfYearsSchedule(cost, salvage, life)
    },
    units = {
      if (is.null(units)) stop("`units` is required for method \"units\".", call. = FALSE)
      unitsSchedule(cost, salvage, units, life)
    }
  )
  attr(out, "method") = method
  out
}

#' Gain or loss on disposal
#'
#' A sale above book value produces a gain, credited to `gainAccount`; a
#' sale below book value mirrors this with a loss, debited to
#' `lossAccount`.
#'
#' @param schedule A depreciation schedule as returned by
#'   [depreciationSchedule()].
#' @param period The period at whose end the asset is sold.
#' @param proceeds Sale proceeds.
#' @param gainAccount,lossAccount Account codes for the gain and the loss.
#' @return A list with elements `bookValue`, `proceeds`, `gain`, `loss`
#'   and `entry` (a data frame with columns `account`, `debit`, `credit`
#'   describing the gain or loss line).
#' @export
disposalResult = function(schedule, period, proceeds, gainAccount = "4900",
                          lossAccount = "6900") {
  if (period < 1 || period > nrow(schedule)) {
    stop("`period` must lie within the schedule.", call. = FALSE)
  }
  bookValue = schedule$closing[schedule$period == period]
  difference = proceeds - bookValue

  if (difference >= 0) {
    gain = difference
    loss = 0
    entry = data.frame(account = gainAccount, debit = 0, credit = gain)
  } else {
    gain = 0
    loss = -difference
    entry = data.frame(account = lossAccount, debit = loss, credit = 0)
  }
  list(bookValue = bookValue, proceeds = proceeds, gain = gain, loss = loss,
    entry = entry)
}

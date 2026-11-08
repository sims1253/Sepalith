#' Check a roster against working-time rules
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param maxConsecutive Most working days in a row.
#' @param maxNights Most night shifts in a row.
#' @param maxHours Most paid hours in any 7-day window.
#' @param hours Paid hours of each shift code.
#' @return A data frame with columns `rule`, `staff`, `value` and `limit`,
#'   one row per staff member breaking a rule.
#' @export
checkRestRules = function(roster, maxConsecutive = 6, maxNights = 4, maxHours = 48,
                          hours = c(E = 8, L = 8, N = 10)) {
  longestRun = function(flags) {
    runs = rle(flags)
    max(c(0, runs$lengths[runs$values]))
  }
  found = list()

  # Consecutive working days
  consecutive = apply(roster != "O", 1, longestRun)
  bad = names(consecutive)[consecutive > maxConsecutive]
  if (length(bad) > 0) {
    found$consecutive = data.frame(rule = "consecutive", staff = bad,
      value = unname(consecutive[bad]), limit = maxConsecutive)
  }
  # Consecutive night shifts
  nights = apply(roster == "N", 1, longestRun)
  bad = names(nights)[nights > maxNights]
  if (length(bad) > 0) {
    found$nights = data.frame(rule = "nights", staff = bad,
      value = unname(nights[bad]), limit = maxNights)
  }
  # Hours in the busiest 7-day window
  paid = matrix(c(hours, O = 0)[roster], nrow = nrow(roster), dimnames = dimnames(roster))
  window = min(7, ncol(roster))
  weekly = apply(paid, 1, function(x) {
    max(stats::filter(x, rep(1, window), sides = 1), na.rm = TRUE)
  })
  bad = names(weekly)[weekly > maxHours]
  if (length(bad) > 0) {
    found$hours = data.frame(rule = "hours", staff = bad,
      value = unname(weekly[bad]), limit = maxHours)
  }
  # Quick returns: an early shift straight after a night or a late
  if (ncol(roster) > 1) {
    quick = rowSums(roster[, -ncol(roster), drop = FALSE] %in% c("N", "L") &
      roster[, -1, drop = FALSE] == "E")
    names(quick) = rownames(roster)
  } else {
    quick = stats::setNames(rep(0, nrow(roster)), rownames(roster))
  }
  bad = names(quick)[quick > 0]
  if (length(bad) > 0) {
    found$quickReturn = data.frame(rule = "quickReturn", staff = bad,
      value = unname(quick[bad]), limit = 0)
  }

  out = do.call(rbind, c(list(data.frame(rule = character(0), staff = character(0),
    value = numeric(0), limit = numeric(0))), unname(found)))
  rownames(out) = NULL
  out
}

#' Spread of unpopular work across staff
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param weekendDays Column positions that fall on a weekend.
#' @param hours Paid hours of each shift code.
#' @return A data frame with one row per measure `nights`, `lates`,
#'   `weekends` and `hours`, and columns `min`, `max`, `range` and `cv`.
#' @export
fairnessReport = function(roster, weekendDays = integer(0), hours = c(E = 8, L = 8, N = 10)) {
  measures = c("nights", "lates", "weekends", "hours")
  spread = function(x) {
    c(min = min(x), max = max(x), range = max(x) - min(x),
      cv = if (mean(x) > 0) stats::sd(x) / mean(x) else 0)
  }

  # Night shifts per person
  nights = rowSums(roster == "N")
  nightsSpread = spread(nights)
  # Late shifts per person
  lates = rowSums(roster == "L")
  latesSpread = spread(lates)
  # Weekend shifts per person
  weekends = rowSums(roster[, weekendDays, drop = FALSE] != "O")
  weekendsSpread = spread(weekends)
  # Paid hours per person
  paidHours = rowSums(matrix(c(hours, O = 0)[roster], nrow = nrow(roster)))
  hoursSpread = spread(paidHours)

  out = as.data.frame(rbind(nightsSpread, latesSpread, weekendsSpread, hoursSpread))
  out = cbind(measure = measures, out)
  rownames(out) = NULL
  # Staff carrying the most nights is the usual first complaint
  attr(out, "mostNights") = rownames(roster)[nights == max(nights)]
  out
}

#' Swap the shifts of two staff members on one day
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param a,b Names of the two staff members.
#' @param day Column index or name of the day.
#' @return The roster with the two cells exchanged; stops when the swap would
#'   put either person on an early shift straight after a night.
#' @export
swapShifts = function(roster, a, b, day) {
  if (is.character(day)) day = match(day, colnames(roster))
  if (is.na(day) || day < 1 || day > ncol(roster)) {
    stop("`day` is not a day of the roster.", call. = FALSE)
  }

  # Check the first person
  if (!a %in% rownames(roster)) {
    stop(sprintf("'%s' is not on the roster.", a), call. = FALSE)
  }
  shiftA = roster[a, day]
  # Check the second person
  if (!b %in% rownames(roster)) {
    stop(sprintf("'%s' is not on the roster.", b), call. = FALSE)
  }
  shiftB = roster[b, day]
  if (shiftA == shiftB) return(roster)

  roster[a, day] = shiftB
  roster[b, day] = shiftA
  # Rest before and after the day for the first person
  if (day > 1 && roster[a, day - 1] == "N" && roster[a, day] == "E") {
    stop(sprintf("Swap leaves '%s' without rest after a night.", a), call. = FALSE)
  }
  if (day < ncol(roster) && roster[a, day] == "N" && roster[a, day + 1] == "E") {
    stop(sprintf("Swap leaves '%s' without rest after a night.", a), call. = FALSE)
  }
  # Rest before and after the day for the second person
  if (day > 1 && roster[b, day - 1] == "N" && roster[b, day] == "E") {
    stop(sprintf("Swap leaves '%s' without rest after a night.", b), call. = FALSE)
  }
  if (day < ncol(roster) && roster[b, day] == "N" && roster[b, day + 1] == "E") {
    stop(sprintf("Swap leaves '%s' without rest after a night.", b), call. = FALSE)
  }
  roster
}

#' Apply availability requests to a roster
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param requests A data frame with columns `staff`, `day` and `type`, and
#'   for preferences a `shift` column.
#' @return The roster with requests applied: `"A"` for annual leave, `"T"`
#'   for training, `"S"` for sickness and preferred shifts where the person was
#'   rostered on another shift. A `released` attribute lists the cells that
#'   were taken off a shift.
#' @export
applyAvailability = function(roster, requests) {
  released = data.frame(staff = character(0), day = character(0), shift = character(0))
  for (i in seq_len(nrow(requests))) {
    who = requests$staff[i]
    day = requests$day[i]
    before = roster[who, day]

    roster[who, day] = switch(requests$type[i],
      leave = {
        # Annual leave always wins over the roster
        "A"
      },
      training = {
        # Training replaces a working shift but not a day off
        if (before == "O") "O" else "T"
      },
      sick = {
        # Sickness replaces whatever was planned
        "S"
      },
      preference = {
        # A preferred shift only replaces another working shift
        if (before == "O") "O" else requests$shift[i]
      },
      stop(sprintf("Unknown request type '%s'.", requests$type[i]), call. = FALSE)
    )

    if (before %in% c("E", "L", "N") && roster[who, day] != before) {
      released = rbind(released, data.frame(staff = who, day = day, shift = before))
    }
  }
  attr(roster, "released") = released
  roster
}

#' Working-time limits for a contract type
#'
#' @param contract Contract type, one of `"full"`, `"part"`, `"casual"` or
#'   `"nightOnly"`.
#' @param base Limits of a full-time contract: `maxHours`, `maxConsecutive`,
#'   `maxNights` and `minDaysOff`.
#' @return A list with elements `maxHours`, `maxConsecutive`, `maxNights`
#'   and `minDaysOff` for the contract.
#' @export
contractLimits = function(contract = c("full", "part", "casual", "nightOnly"),
                          base = list(maxHours = 48, maxConsecutive = 6, maxNights = 4,
                                      minDaysOff = 2)) {
  contract = match.arg(contract)

  limits = switch(contract,
    full = {
      # Full-time staff work to the base limits
      base
    },
    part = {
      # Part-time staff work half the hours, other limits unchanged
      out = base
      out$maxHours = base$maxHours / 2
      out
    },
    casual = {
      # Casual staff work short runs with extra days off
      out = base
      out$maxConsecutive = base$maxConsecutive / 2
      out$minDaysOff = base$minDaysOff + 1
      out
    },
    nightOnly = {
      # Night staff may work nights up to the consecutive-day limit
      out = base
      out$maxNights = base$maxConsecutive
      out
    }
  )

  if (any(unlist(limits) < 0)) {
    stop("Limits must not be negative.", call. = FALSE)
  }
  limits
}

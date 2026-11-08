#' Staffing level of every shift on every day
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param early,late,night Required head count of each shift.
#' @return A data frame with columns `day`, `shift`, `staffed`, `required`
#'   and `gap` (negative when short), ordered by day and shift.
#' @export
coverageByShift = function(roster, early = 2, late = 2, night = 1) {
  days = colnames(roster)
  if (is.null(days)) days = paste0("d", seq_len(ncol(roster)))

  # Early shift
  earlyStaffed = colSums(roster == "E")
  earlyRows = data.frame(day = days, shift = "E", staffed = unname(earlyStaffed),
    required = early)
  earlyRows$gap = earlyRows$staffed - earlyRows$required
  # Late shift
  lateStaffed = colSums(roster == "L")
  lateRows = data.frame(day = days, shift = "L", staffed = unname(lateStaffed),
    required = late)
  lateRows$gap = lateRows$staffed - lateRows$required
  # Night shift
  nightStaffed = colSums(roster == "N")
  nightRows = data.frame(day = days, shift = "N", staffed = unname(nightStaffed),
    required = night)
  nightRows$gap = nightRows$staffed - nightRows$required

  out = rbind(earlyRows, lateRows, nightRows)
  out = out[order(match(out$day, days), match(out$shift, c("E", "L", "N"))), ]
  rownames(out) = NULL
  out
}

#' Select coverage rows that need attention
#'
#' @param coverage A data frame as returned by [coverageByShift()].
#' @param type Which rows to keep, one of `"shortfall"`, `"surplus"`,
#'   `"both"` or `"critical"`.
#' @param critical Shortfall (in heads) from which a gap is critical.
#' @return The selected rows with an added `status` column.
#' @export
coverageGaps = function(coverage, type = c("shortfall", "surplus", "both", "critical"),
                        critical = 2) {
  type = match.arg(type)

  keep = switch(type,
    shortfall = {
      # Fewer staff than required
      flag = coverage$gap < 0
      flag
    },
    surplus = {
      # More staff than required
      flag = coverage$gap > 0
      flag
    },
    both = {
      # Any deviation from the requirement
      flag = coverage$gap != 0
      flag
    },
    critical = {
      # Shortfalls at or beyond the critical head count, or an empty shift
      flag = coverage$gap <= -critical | (coverage$staffed == 0 & coverage$required > 0)
      flag
    }
  )

  out = coverage[keep, , drop = FALSE]
  out$status = ifelse(out$gap < 0, "short", ifelse(out$gap > 0, "over", "ok"))
  rownames(out) = NULL
  out
}

#' Fill shortfalls from staff who are off
#'
#' @param roster A character matrix with staff in rows, days in columns and
#'   shift codes `"E"`, `"L"`, `"N"` or `"O"`.
#' @param early,late,night Required head count of each shift.
#' @param maxShifts Most shifts any staff member may work over the roster.
#' @return The updated roster with an `added` attribute counting the extra
#'   assignments.
#' @export
fillGaps = function(roster, early = 2, late = 2, night = 1, maxShifts = Inf) {
  added = 0
  pickSpare = function(day, needed, blocked) {
    spare = roster[, day] == "O" & rowSums(roster != "O") < maxShifts
    if (day > 1) spare = spare & !(roster[, day - 1] %in% blocked)
    candidates = which(spare)
    candidates = candidates[order(rowSums(roster != "O")[candidates])]
    utils::head(candidates, needed)
  }

  for (d in seq_len(ncol(roster))) {
    # Early shift, never straight after a night
    missing = early - sum(roster[, d] == "E")
    if (missing > 0) {
      who = pickSpare(d, missing, blocked = "N")
      roster[who, d] = "E"
      added = added + length(who)
    }
    # Late shift, any spare staff member may take it
    missing = late - sum(roster[, d] == "L")
    if (missing > 0) {
      who = pickSpare(d, missing, blocked = character(0))
      roster[who, d] = "L"
      added = added + length(who)
    }
    # Night shift, any spare staff member may take it
    missing = night - sum(roster[, d] == "N")
    if (missing > 0) {
      who = pickSpare(d, missing, blocked = character(0))
      roster[who, d] = "N"
      added = added + length(who)
    }
  }
  attr(roster, "added") = added
  roster
}

#' Summarise coverage per shift
#'
#' @param coverage A data frame as returned by [coverageByShift()].
#' @return A data frame with one row per shift `"E"`, `"L"` and `"N"` and
#'   columns `daysShort`, `headsShort`, `daysOver`, `headsOver`, `worstDay`
#'   and `fillRate`.
#' @export
coverageSummary = function(coverage) {
  shifts = c("E", "L", "N")
  out = data.frame(
    shift = shifts,
    daysShort = NA_integer_,
    headsShort = NA_real_,
    daysOver = NA_integer_,
    headsOver = NA_real_,
    worstDay = NA_character_,
    fillRate = NA_real_
  )

  for (i in seq_along(shifts)) {
    rows = coverage[coverage$shift == shifts[i], , drop = FALSE]
    out$daysShort[i] = sum(rows$gap < 0)
    out$headsShort[i] = -sum(pmin(rows$gap, 0))
    out$daysOver[i] = sum(rows$gap > 0)
    out$headsOver[i] = sum(pmax(rows$gap, 0))
    out$worstDay[i] = if (any(rows$gap < 0)) rows$day[which.min(rows$gap)] else NA_character_
    out$fillRate[i] = sum(pmin(rows$staffed, rows$required)) / sum(rows$required)
  }

  # A shift without any requirement counts as fully filled
  out$fillRate[is.nan(out$fillRate)] = 1
  out
}

#' Check skill coverage of the staff on one shift
#'
#' @param onShift Character vector of staff names working the shift.
#' @param skills A data frame with a `staff` column and one logical column per
#'   skill `forklift`, `firstAid`, `supervisor` and `qa`.
#' @param needs Minimum number of holders of each skill on the shift.
#' @return A data frame with columns `skill`, `needed`, `present`, `holders`
#'   and `met`.
#' @export
skillCoverage = function(onShift, skills,
                         needs = c(forklift = 1, firstAid = 1, supervisor = 1, qa = 0)) {
  crew = skills[skills$staff %in% onShift, , drop = FALSE]
  unknown = setdiff(onShift, skills$staff)
  if (length(unknown) > 0) {
    warning("No skill record for: ", paste(unknown, collapse = ", "), call. = FALSE)
  }

  # Forklift licence
  forkliftHolders = crew$staff[crew$forklift]
  forkliftMet = length(forkliftHolders) >= needs[["forklift"]]
  # First aid certificate
  firstAidHolders = crew$staff[crew$firstAid]
  firstAidMet = length(firstAidHolders) >= needs[["firstAid"]]
  # Shift supervisor
  supervisorHolders = crew$staff[crew$supervisor]
  supervisorMet = length(supervisorHolders) >= needs[["supervisor"]]
  # Quality inspection
  qaHolders = crew$staff[crew$qa]
  qaMet = length(qaHolders) >= needs[["qa"]]

  data.frame(
    skill = c("forklift", "firstAid", "supervisor", "qa"),
    needed = unname(needs[c("forklift", "firstAid", "supervisor", "qa")]),
    present = c(length(forkliftHolders), length(firstAidHolders),
                length(supervisorHolders), length(qaHolders)),
    holders = c(paste(forkliftHolders, collapse = ", "), paste(firstAidHolders, collapse = ", "),
                paste(supervisorHolders, collapse = ", "), paste(qaHolders, collapse = ", ")),
    met = c(forkliftMet, firstAidMet, supervisorMet, qaMet)
  )
}

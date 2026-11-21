#' Clock windows of the shifts on one day
#'
#' @param date A date or a string in `YYYY-MM-DD` form.
#' @param early,late,night Start and end hour of each shift, counted from
#'   midnight of `date`; hours above 24 run into the next day.
#' @param tz Time zone of the returned times.
#' @return A data frame with columns `shift`, `start`, `end` and `hours`.
#' @export
shiftWindows = function(date, early = c(6, 14), late = c(14, 22), night = c(22, 30),
                        tz = "UTC") {
  midnight = as.POSIXct(paste(as.Date(date), "00:00:00"), tz = tz)

  # Early shift
  if (length(early) != 2 || early[2] <= early[1]) {
    stop(sprintf("`%s` must be an increasing pair of hours.", "early"), call. = FALSE)
  }
  earlyStart = midnight + early[1] * 3600
  earlyEnd = midnight + early[2] * 3600
  # Late shift
  if (length(late) != 2 || late[2] <= late[1]) {
    stop(sprintf("`%s` must be an increasing pair of hours.", "late"), call. = FALSE)
  }
  lateStart = midnight + late[1] * 3600
  lateEnd = midnight + late[2] * 3600
  # Night shift
  if (length(night) != 2 || night[2] <= night[1]) {
    stop(sprintf("`%s` must be an increasing pair of hours.", "night"), call. = FALSE)
  }
  nightStart = midnight + night[1] * 3600
  nightEnd = midnight + night[2] * 3600

  # Overlapping windows usually mean a typo in the hours
  if (earlyEnd > lateStart || lateEnd > nightStart) {
    warning("Shift windows overlap.", call. = FALSE)
  }
  data.frame(
    shift = c("early", "late", "night"),
    start = c(earlyStart, lateStart, nightStart),
    end = c(earlyEnd, lateEnd, nightEnd),
    hours = c(diff(early), diff(late), diff(night))
  )
}

#' Shift sequence of one rotation pattern
#'
#' @param days Number of days to generate.
#' @param pattern Rotation pattern, one of `"fixed"`, `"forward"`,
#'   `"backward"`, `"weekly"` or `"split"`.
#' @param shifts Shift codes in forward order.
#' @param offRun Number of days off between blocks.
#' @param blockLength Number of consecutive days in one block.
#' @return A character vector of length `days` with shift codes and `"O"`
#'   for days off.
#' @export
rotationPattern = function(days, pattern = c("fixed", "forward", "backward", "weekly", "split"),
                           shifts = c("E", "L", "N"), offRun = 2, blockLength = 5) {
  pattern = match.arg(pattern)
  off = rep("O", offRun)

  cycle = switch(pattern,
    fixed = {
      # Same shift in every block
      block = rep(shifts[1], blockLength)
      c(block, off)
    },
    forward = {
      # Each block moves one shift later
      blocks = lapply(shifts, function(code) c(rep(code, blockLength), off))
      unlist(blocks)
    },
    backward = {
      # Each block moves one shift earlier
      blocks = lapply(rev(shifts), function(code) c(rep(code, blockLength), off))
      unlist(blocks)
    },
    weekly = {
      # Each shift is held for a calendar week, days off at the weekend
      blocks = lapply(shifts, function(code) c(rep(code, 7 - offRun), off))
      unlist(blocks)
    },
    split = {
      # Alternate the first and last shift inside each block
      block = rep(c(shifts[1], shifts[length(shifts)]), length.out = blockLength)
      c(block, off)
    }
  )

  rep(cycle, length.out = days)
}

#' Assign staff to shift blocks day by day
#'
#' @param staff Character vector of staff names.
#' @param days Number of days to roster.
#' @param early,late,night Number of staff needed on each shift per day.
#' @return A character matrix with one row per staff member and one column
#'   per day holding `"E"`, `"L"`, `"N"` or `"O"`.
#' @export
assignRosterBlocks = function(staff, days, early = 2, late = 2, night = 1) {
  if (early + late + night > length(staff)) {
    stop("Not enough staff to cover one day.", call. = FALSE)
  }
  roster = matrix("O", nrow = length(staff), ncol = days,
    dimnames = list(staff, paste0("d", seq_len(days))))
  pickStaff = function(day, blocked) {
    free = roster[, day] == "O"
    if (day > 1) free = free & !(roster[, day - 1] %in% blocked)
    if (!any(free)) stop(sprintf("No staff left on day %d.", day), call. = FALSE)
    worked = rowSums(roster != "O")
    candidates = which(free)
    candidates[which.min(worked[candidates])]
  }

  for (d in seq_len(days)) {
    # Night shift first, any staff member may take it
    for (k in seq_len(night)) {
      who = pickStaff(d, blocked = character(0))
      roster[who, d] = "N"
    }
    # Early shift, never straight after a night
    for (k in seq_len(early)) {
      who = pickStaff(d, blocked = "N")
      roster[who, d] = "E"
    }
    # Late shift, any staff member may take it
    for (k in seq_len(late)) {
      who = pickStaff(d, blocked = character(0))
      roster[who, d] = "L"
    }
  }
  roster
}

#' Worked shifts and hours per staff member
#'
#' @param roster A character matrix as returned by [assignRosterBlocks()].
#' @param hours Paid hours of each shift code.
#' @return A data frame with columns `staff`, `early`, `late`, `night`,
#'   `shifts` and `hours`.
#' @export
rosterHours = function(roster, hours = c(E = 8, L = 8, N = 10)) {
  # Early shifts
  earlyCount = rowSums(roster == "E")
  earlyHours = earlyCount * hours[["E"]]
  # Late shifts
  lateCount = rowSums(roster == "L")
  lateHours = lateCount * hours[["L"]]
  # Night shifts
  nightCount = rowSums(roster == "N")
  nightHours = nightCount * hours[["N"]]

  out = data.frame(
    staff = rownames(roster),
    early = unname(earlyCount),
    late = unname(lateCount),
    night = unname(nightCount),
    shifts = unname(earlyCount + lateCount + nightCount),
    hours = unname(earlyHours + lateHours + nightHours)
  )
  rownames(out) = NULL
  out
}

#' Consecutive blocks of the same shift
#'
#' @param roster A character matrix as returned by [assignRosterBlocks()].
#' @param codes Shift codes to report; days off are never reported.
#' @return A data frame with columns `staff`, `shift`, `first`, `last` and
#'   `length`, one row per block.
#' @export
rosterBlocks = function(roster, codes = c("E", "L", "N")) {
  rows = vector("list", nrow(roster))
  for (i in seq_len(nrow(roster))) {
    runs = rle(roster[i, ])
    last = cumsum(runs$lengths)
    first = last - runs$lengths + 1
    keep = runs$values %in% codes
    rows[[i]] = data.frame(
      staff = rep(rownames(roster)[i], sum(keep)),
      shift = runs$values[keep],
      first = first[keep],
      last = last[keep],
      length = runs$lengths[keep]
    )
  }
  out = do.call(rbind, rows)
  rownames(out) = NULL
  out
}

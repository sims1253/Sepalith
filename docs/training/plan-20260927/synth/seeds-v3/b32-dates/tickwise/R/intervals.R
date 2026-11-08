#' @title Create time intervals
#' @description Builds a data frame of half-open intervals `[start, end)`
#'   stored as `POSIXct` in a single time zone.
#' @param start,end Vectors coercible to `POSIXct`, recycled to a common
#'   length.
#' @param tz Time zone of the stored times.
#' @return A data frame of class `tw_interval` with columns `start` and
#'   `end`.
#' @export
interval.new <- function(start, end, tz = "UTC") {
  start <- as.POSIXct(start, tz = tz)
  end <- as.POSIXct(end, tz = tz)
  n <- max(length(start), length(end))

  # Check `start`
  if (anyNA(start)) {
    stop("`start` must not contain missing values.", call. = FALSE)
  }
  # Check `end`
  if (anyNA(end)) {
    stop("`end` must not contain missing values.", call. = FALSE)
  }
  # Check the order of `start` and `end`
  start <- rep_len(start, n)
  end <- rep_len(end, n)
  if (any(end < start)) {
    stop("`end` must not be before `start`.", call. = FALSE)
  }
  out <- data.frame(start = start, end = end)
  class(out) <- c("tw_interval", "data.frame")
  out
}

#' @title Interval lengths
#' @param iv A `tw_interval`.
#' @param units Units passed to [difftime()].
#' @return A `difftime` vector.
#' @export
interval.length <- function(iv, units = c("hours", "secs", "mins", "days", "weeks")) {
  units <- match.arg(units)
  difftime(iv$end, iv$start, units = units)
}

#' @title Grow or shrink intervals on either side
#' @param iv A `tw_interval`.
#' @param before,after `difftime` (or seconds) to move `start` earlier and
#'   `end` later; negative values shrink the interval.
#' @return A `tw_interval`; an error if an interval would become negative.
#' @export
interval.extend <- function(iv, before = 0, after = 0) {
  before <- as.numeric(before, units = "secs")
  after <- as.numeric(after, units = "secs")
  iv$start <- iv$start - before
  iv$end <- iv$end + after
  if (any(iv$end < iv$start)) {
    stop("`end` must not be before `start`.", call. = FALSE)
  }
  iv
}

#' @title Pairwise intersection of two sets of intervals
#' @param a,b `tw_interval` objects of the same length.
#' @return A data frame with columns `start`, `end` and `overlap` (seconds
#'   of overlap); `start` and `end` are `NA` where the pair does not
#'   overlap.
#' @export
interval.overlap <- function(a, b) {
  if (nrow(a) != nrow(b)) {
    stop("`a` and `b` must have the same number of intervals.", call. = FALSE)
  }
  later.start <- a$start
  take.b <- b$start > a$start
  later.start[take.b] <- b$start[take.b]
  earlier.end <- a$end
  take.b <- b$end < a$end
  earlier.end[take.b] <- b$end[take.b]

  overlap <- pmax(as.numeric(earlier.end) - as.numeric(later.start), 0)
  later.start[overlap == 0] <- NA
  earlier.end[overlap == 0] <- NA
  data.frame(start = later.start, end = earlier.end, overlap = overlap)
}

#' @title Split intervals at calendar boundaries
#' @param iv A `tw_interval` with a single row.
#' @param by One of `"hour"`, `"day"`, `"week"` or `"month"`.
#' @return A `tw_interval` whose pieces cover `iv` exactly, each lying
#'   within one `by` period.
#' @export
interval.split <- function(iv, by = c("hour", "day", "week", "month")) {
  by <- match.arg(by)
  if (nrow(iv) != 1L) {
    stop("`iv` must contain exactly one interval.", call. = FALSE)
  }
  tz <- attr(iv$start, "tzone")
  lt <- as.POSIXlt(iv$start, tz = tz)

  first <- switch(by,
    hour = {
      lt$min <- 0
      lt$sec <- 0
      as.POSIXct(lt)
    },
    day = {
      as.POSIXct(format(lt, "%Y-%m-%d"), tz = tz)
    },
    week = {
      monday <- as.Date(format(lt, "%Y-%m-%d")) - (as.integer(format(lt, "%u")) - 1L)
      as.POSIXct(format(monday), tz = tz)
    },
    month = {
      as.POSIXct(format(lt, "%Y-%m-01"), tz = tz)
    }
  )
  cuts <- seq(first, by = paste("1", by), length.out = 2L)
  while (cuts[length(cuts)] < iv$end) {
    cuts <- seq(first, by = paste("1", by), length.out = length(cuts) * 2L)
  }
  cuts <- cuts[cuts > iv$start & cuts < iv$end]
  bounds <- c(iv$start, cuts, iv$end)
  interval.new(bounds[-length(bounds)], bounds[-1L], tz = tz)
}

#' @title Gaps between intervals
#' @param iv A `tw_interval`.
#' @param min.gap Gaps shorter than this many seconds are ignored.
#' @return A `tw_interval` of the uncovered stretches between the earliest
#'   `start` and the latest `end`.
#' @keywords internal
interval.gaps <- function(iv, min.gap = 0) {
  tz <- attr(iv$start, "tzone")
  iv <- iv[order(iv$start), , drop = FALSE]
  covered.until <- iv$end[1L]
  gap.start <- iv$start[0L]
  gap.end <- iv$end[0L]
  for (i in seq_len(nrow(iv))[-1L]) {
    if (iv$start[i] > covered.until) {
      gap.start <- c(gap.start, covered.until)
      gap.end <- c(gap.end, iv$start[i])
    }
    if (iv$end[i] > covered.until) {
      covered.until <- iv$end[i]
    }
  }
  keep <- as.numeric(gap.end) - as.numeric(gap.start) >= min.gap
  interval.new(gap.start[keep], gap.end[keep], tz = tz)
}

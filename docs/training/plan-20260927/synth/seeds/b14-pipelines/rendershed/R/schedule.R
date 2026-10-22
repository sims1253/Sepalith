weekdayCodes <- c("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#' Combine a date with a cadence's time of day
#'
#' @param date A `Date`.
#' @param cadence A cadence created by [parseCadence()].
#' @return A `POSIXct` in the cadence's time zone.
#' @noRd
atTimeOfDay <- function(date, cadence) {
  stamp <- sprintf("%s %02d:%02d:00", format(date), cadence$hour, cadence$minute)
  as.POSIXct(stamp, tz = cadence$tz)
}

#' Parse a report cadence specification
#'
#' Cadences are short strings such as `"daily@07:30"`, `"weekly:mon@08:00"`
#' or `"monthly:1@06:00"`. Monthly anchors are limited to days 1 to 28.
#'
#' @param spec A single cadence string.
#' @param tz The time zone the time of day refers to.
#' @return An object of class `rendershed_cadence`.
#' @family scheduling
#' @export
parseCadence <- function(spec, tz = "UTC") {
  spec <- trimws(tolower(spec))
  parts <- strsplit(spec, "@", fixed = TRUE)[[1]]
  if (length(parts) != 2L) {
    stop("Cadence must look like 'daily@HH:MM', got '", spec, "'.", call. = FALSE)
  }
  head <- strsplit(parts[[1]], ":", fixed = TRUE)[[1]]
  clock <- suppressWarnings(as.integer(strsplit(parts[[2]], ":", fixed = TRUE)[[1]]))
  if (length(clock) != 2L || anyNA(clock) || clock[1] > 23L || clock[2] > 59L) {
    stop("Invalid time of day in cadence '", spec, "'.", call. = FALSE)
  }
  type <- match.arg(head[[1]], c("daily", "weekly", "monthly"))
  anchor <- if (length(head) > 1L) head[[2]] else NA_character_
  if (type == "weekly" && !anchor %in% weekdayCodes) {
    stop("Weekly cadence needs a weekday such as 'mon', got '", anchor, "'.", call. = FALSE)
  }
  if (type == "monthly" && !anchor %in% as.character(1:28)) {
    stop("Monthly cadence needs a day between 1 and 28, got '", anchor, "'.", call. = FALSE)
  }
  structure(
    list(type = type, anchor = anchor, hour = clock[1], minute = clock[2], tz = tz),
    class = "rendershed_cadence"
  )
}

#' Next run time for a daily cadence
#'
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @keywords internal
nextRunDaily <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)
  candidate <- atTimeOfDay(today, cadence)
  if (candidate <= from) {
    candidate <- atTimeOfDay(today + 1L, cadence)
  }
  candidate
}

#' Next run time for a weekly cadence
#'
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @keywords internal
nextRunWeekly <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)
  target <- match(cadence$anchor, weekdayCodes)
  current <- as.integer(format(today, "%u"))
  offset <- (target - current) %% 7L
  candidate <- atTimeOfDay(today + offset, cadence)
  if (candidate <= from) {
    candidate <- atTimeOfDay(today + offset + 7L, cadence)
  }
  candidate
}

#' Next run time for a monthly cadence
#'
#' @param cadence A cadence created by [parseCadence()].
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @keywords internal
nextRunMonthly <- function(cadence, from = Sys.time()) {
  today <- as.Date(from, tz = cadence$tz)
  day <- as.integer(cadence$anchor)
  firstOfMonth <- as.Date(format(today, "%Y-%m-01"))
  candidate <- atTimeOfDay(firstOfMonth + day - 1L, cadence)
  if (candidate <= from) {
    nextMonth <- seq(firstOfMonth, by = "1 month", length.out = 2L)[2L]
    candidate <- atTimeOfDay(nextMonth + day - 1L, cadence)
  }
  candidate
}

#' Next run time for any cadence
#'
#' Dispatches to the daily, weekly or monthly helper.
#'
#' @param cadence A cadence created by [parseCadence()], or a cadence string.
#' @param from The reference time; the result is strictly after it.
#' @return A `POSIXct` time stamp.
#' @family scheduling
#' @export
nextRun <- function(cadence, from = Sys.time()) {
  if (is.character(cadence)) {
    cadence <- parseCadence(cadence)
  }
  switch(cadence$type,
    daily = nextRunDaily(cadence, from),
    weekly = nextRunWeekly(cadence, from),
    monthly = nextRunMonthly(cadence, from)
  )
}

#' Is a report due?
#'
#' A report that has never run is always due. Otherwise it is due once the
#' next run after `lastRun` has passed.
#'
#' @param cadence A cadence created by [parseCadence()], or a cadence string.
#' @param lastRun Time of the last successful run, or `NA`.
#' @param now The current time.
#' @return `TRUE` or `FALSE`.
#' @family scheduling
#' @export
isDue <- function(cadence, lastRun, now = Sys.time()) {
  if (length(lastRun) == 0L || is.na(lastRun)) {
    return(TRUE)
  }
  nextRun(cadence, from = lastRun) <= now
}

#' Read a report registry
#'
#' The registry is a YAML file with a top-level `reports` list. Each entry
#' needs an `id`, an `input` document and a `cadence`; `format`,
#' `recipients` and `tz` are optional.
#'
#' @param path Path to the registry YAML file.
#' @return A data frame with one row per report.
#' @family scheduling
#' @export
readRegistry <- function(path) {
  raw <- yaml::read_yaml(path)
  if (is.null(raw$reports) || length(raw$reports) == 0L) {
    stop("Registry '", path, "' does not list any reports.", call. = FALSE)
  }
  required <- c("id", "input", "cadence")
  rows <- lapply(raw$reports, function(entry) {
    missing <- setdiff(required, names(entry))
    if (length(missing) > 0L) {
      stop("Report entry is missing: ", paste(missing, collapse = ", "), call. = FALSE)
    }
    data.frame(
      id = entry$id,
      input = normalizePath(file.path(dirname(path), entry$input), mustWork = FALSE),
      cadence = entry$cadence,
      format = entry$format %||% "html",
      recipients = paste(unlist(entry$recipients), collapse = "; "),
      tz = entry$tz %||% "UTC",
      stringsAsFactors = FALSE
    )
  })
  registry <- do.call(rbind, rows)
  if (anyDuplicated(registry$id)) {
    stop("Duplicate report ids in registry: ", path, call. = FALSE)
  }
  registry
}

#' Select the reports that are due now
#'
#' Joins the registry with the last successful run per report and keeps
#' the rows whose cadence says they should run.
#'
#' @param registry A data frame returned by [readRegistry()].
#' @param lastRuns A data frame with columns `id` and `lastRun`.
#' @param now The current time.
#' @return The subset of `registry` that is due, with a `lastRun` column.
#' @family scheduling
#' @export
dueReports <- function(registry, lastRuns, now = Sys.time()) {
  merged <- registry |>
    merge(lastRuns, by = "id", all.x = TRUE, sort = FALSE)
  due <- vapply(seq_len(nrow(merged)), function(i) {
    cadence <- parseCadence(merged$cadence[i], tz = merged$tz[i])
    isDue(cadence, merged$lastRun[i], now = now)
  }, logical(1))
  merged[due, , drop = FALSE] |>
    (\(d) d[order(d$id), , drop = FALSE])()
}

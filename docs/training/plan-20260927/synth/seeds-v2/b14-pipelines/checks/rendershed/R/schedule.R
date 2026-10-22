# skipped: needs yaml (readRegistry)

at <- function(x) as.POSIXct(x, tz = "UTC")

# parseCadence()
weekly <- parseCadence("Weekly:FRI@08:00")
stopifnot(
  inherits(weekly, "rendershed_cadence"),
  identical(weekly$anchor, "fri"),
  identical(c(weekly$hour, weekly$minute), c(8L, 0L)),
  inherits(try(parseCadence("daily@-1:30"), silent = TRUE), "try-error"),
  inherits(try(parseCadence("monthly:29@06:00"), silent = TRUE), "try-error"),
  inherits(try(parseCadence(NA_character_), silent = TRUE), "try-error")
)

# atTimeOfDay()
stopifnot(identical(atTimeOfDay(as.Date("2026-03-02"), weekly), at("2026-03-02 08:00:00")))

# nextRunDaily(), nextRunWeekly(), nextRunMonthly(), nextRun()
stopifnot(
  identical(nextRunDaily(parseCadence("daily@07:30"), at("2026-03-02 07:30:00")),
            at("2026-03-03 07:30:00")),
  identical(nextRunWeekly(weekly, at("2026-03-02 12:00:00")), at("2026-03-06 08:00:00")),
  identical(nextRunWeekly(weekly, at("2026-03-06 08:00:00")), at("2026-03-13 08:00:00")),
  identical(nextRunMonthly(parseCadence("monthly:28@06:00"), at("2026-02-28 07:00:00")),
            at("2026-03-28 06:00:00")),
  identical(nextRun("daily@07:30", at("2026-12-31 23:00:00")), at("2027-01-01 07:30:00"))
)

# isDue()
stopifnot(
  isTRUE(isDue("daily@07:30", NA)),
  isTRUE(isDue("daily@07:30", at("2026-03-01 07:30:00"), now = at("2026-03-02 08:00:00"))),
  isFALSE(isDue("daily@07:30", at("2026-03-02 07:30:00"), now = at("2026-03-02 08:00:00")))
)

# dueReports()
registry <- data.frame(
  id = c("c", "a", "b"),
  cadence = c("daily@07:00", "daily@07:00", "weekly:mon@07:00"),
  tz = "UTC", stringsAsFactors = FALSE
)
lastRuns <- data.frame(
  id = c("a", "b"),
  lastRun = at(c("2026-03-03 07:00:00", "2026-03-02 07:00:00")),
  cadence = "ignored"
)
due <- dueReports(registry, lastRuns, now = at("2026-03-04 07:15:00"))
stopifnot(identical(due$id, c("a", "c")), "lastRun" %in% names(due))

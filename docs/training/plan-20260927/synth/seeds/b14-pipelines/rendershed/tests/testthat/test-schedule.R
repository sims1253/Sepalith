at <- function(x) as.POSIXct(x, tz = "UTC")

test_that("parseCadence reads daily, weekly and monthly specs", {
  daily <- parseCadence("daily@07:30")
  weekly <- parseCadence("Weekly:MON@08:00")
  monthly <- parseCadence("monthly:15@06:05")
  expect_s3_class(daily, "rendershed_cadence")
  expect_equal(c(daily$hour, daily$minute), c(7L, 30L))
  expect_equal(weekly$anchor, "mon")
  expect_equal(monthly$anchor, "15")
})

test_that("parseCadence rejects malformed specs", {
  expect_error(parseCadence("daily"), "daily@HH:MM")
  expect_error(parseCadence("daily@25:00"), "Invalid time of day")
  expect_error(parseCadence("weekly:funday@08:00"), "weekday")
  expect_error(parseCadence("monthly:31@08:00"), "between 1 and 28")
})

test_that("nextRun for a daily cadence rolls over to tomorrow", {
  cadence <- parseCadence("daily@07:30")
  expect_equal(nextRun(cadence, at("2026-03-02 06:00:00")), at("2026-03-02 07:30:00"))
  expect_equal(nextRun(cadence, at("2026-03-02 07:30:00")), at("2026-03-03 07:30:00"))
})

test_that("nextRun for a weekly cadence finds the next weekday", {
  cadence <- parseCadence("weekly:fri@08:00")
  # 2026-03-02 is a Monday
  expect_equal(nextRun(cadence, at("2026-03-02 12:00:00")), at("2026-03-06 08:00:00"))
  expect_equal(nextRun(cadence, at("2026-03-06 09:00:00")), at("2026-03-13 08:00:00"))
})

test_that("nextRun for a monthly cadence moves to the next month", {
  cadence <- parseCadence("monthly:1@06:00")
  expect_equal(nextRun(cadence, at("2026-03-01 05:00:00")), at("2026-03-01 06:00:00"))
  expect_equal(nextRun(cadence, at("2026-12-01 06:30:00")), at("2027-01-01 06:00:00"))
})

test_that("isDue treats a missing last run as due", {
  expect_true(isDue("daily@07:30", NA))
  expect_true(isDue("daily@07:30", at("2026-03-01 07:30:00"), now = at("2026-03-02 08:00:00")))
  expect_false(isDue("daily@07:30", at("2026-03-02 07:30:00"), now = at("2026-03-02 08:00:00")))
})

test_that("readRegistry builds one row per report with defaults", {
  dir <- withr::local_tempdir()
  path <- file.path(dir, "registry.yml")
  writeLines(c(
    "reports:",
    "  - id: berths",
    "    input: berths.qmd",
    "    cadence: daily@07:00",
    "    recipients: [ops@example.org, lead@example.org]",
    "  - id: monthly-volume",
    "    input: volume.Rmd",
    "    cadence: monthly:1@06:00",
    "    format: pdf"
  ), path)
  registry <- readRegistry(path)
  expect_equal(registry$id, c("berths", "monthly-volume"))
  expect_equal(registry$format, c("html", "pdf"))
  expect_equal(registry$recipients[1], "ops@example.org; lead@example.org")
})

test_that("dueReports keeps only reports whose cadence has elapsed", {
  registry <- data.frame(
    id = c("a", "b", "c"),
    cadence = c("daily@07:00", "weekly:mon@07:00", "daily@07:00"),
    tz = "UTC",
    stringsAsFactors = FALSE
  )
  lastRuns <- data.frame(
    id = c("a", "b"),
    lastRun = at(c("2026-03-03 07:00:00", "2026-03-02 07:00:00"))
  )
  due <- dueReports(registry, lastRuns, now = at("2026-03-04 07:15:00"))
  expect_equal(due$id, c("a", "c"))
})

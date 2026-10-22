test_that("parseDates accepts mixed formats", {
  x <- c("2024-03-15", "15.03.2024", "03/15/2024", NA)
  out <- parseDates(x)
  expect_s3_class(out, "Date")
  expect_equal(out[1:3], rep(as.Date("2024-03-15"), 3))
  expect_true(is.na(out[4]))
})

test_that("parseDates warns about unparseable values", {
  expect_warning(out <- parseDates(c("2024-01-01", "soon")), "1 value")
  expect_true(is.na(out[2]))
})

test_that("daysBetween and weeksBetween agree", {
  from <- as.Date("2024-01-01")
  to <- as.Date("2024-01-29")
  expect_equal(daysBetween(from, to), 28)
  expect_equal(weeksBetween(from, to), 4)
  expect_equal(daysBetween(to, from), -28)
  expect_equal(weeksBetween(to, from), -4)
})

test_that("monthsBetween only counts complete months", {
  expect_equal(monthsBetween("2024-01-15", "2024-03-14"), 1L)
  expect_equal(monthsBetween("2024-01-15", "2024-03-15"), 2L)
  expect_equal(monthsBetween("2023-11-30", "2024-02-29"), 2L)
})

test_that("monthStarts covers both end months", {
  out <- monthStarts("2024-01-20", "2024-04-02")
  expect_length(out, 4)
  expect_equal(out[1], as.Date("2024-01-01"))
  expect_equal(out[4], as.Date("2024-04-01"))
  expect_error(monthStarts("2024-05-01", "2024-01-01"), "before")
})

test_that("workingDays skips weekends and holidays", {
  days <- workingDays("2024-12-23", "2024-12-29",
                      holidays = as.Date(c("2024-12-25", "2024-12-26")))
  expect_equal(days, as.Date(c("2024-12-23", "2024-12-24", "2024-12-27")))
})

test_that("periodLabel switches on period", {
  d <- as.Date(c("2024-02-10", "2024-11-30"))
  expect_equal(periodLabel(d), c("2024-02", "2024-11"))
  expect_equal(periodLabel(d, "quarter"), c("2024-Q1", "2024-Q4"))
  expect_equal(periodLabel(d, "year"), c("2024", "2024"))
  expect_error(periodLabel(d, "week"))
})

test_that("addMonths clamps to the end of the month", {
  expect_equal(addMonths(as.Date("2024-01-31"), 1), as.Date("2024-02-29"))
  expect_equal(addMonths(as.Date("2023-01-31"), 1), as.Date("2023-02-28"))
  expect_equal(addMonths(as.Date("2024-03-15"), -3), as.Date("2023-12-15"))
  expect_equal(addMonths(as.Date("2024-12-31"), 2), as.Date("2025-02-28"))
})

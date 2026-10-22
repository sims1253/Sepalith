test_that("rollMean matches a naive loop for right alignment", {
  x <- c(4, 8, 15, 16, 23, 42)
  expected <- c(NA, NA, 9, 13, 18, 27)
  expect_equal(rollMean(x, width = 3), expected)
})

test_that("rollMean centre and left alignment shift the window", {
  x <- c(4, 8, 15, 16, 23, 42)
  expect_equal(rollMean(x, width = 3, align = "centre"), c(NA, 9, 13, 18, 27, NA))
  expect_equal(rollMean(x, width = 3, align = "left"), c(9, 13, 18, 27, NA, NA))
})

test_that("rollSum and rollMax agree with base R on full windows", {
  x <- c(2, 7, 1, 8, 2, 8, 1, 8)
  sums <- rollSum(x, width = 4)
  maxes <- rollMax(x, width = 4)

  expect_equal(sums[4:8], vapply(4:8, function(i) sum(x[(i - 3):i]), numeric(1)))
  expect_equal(maxes[4:8], vapply(4:8, function(i) max(x[(i - 3):i]), numeric(1)))
  expect_true(all(is.na(sums[1:3])))
})

test_that("rollSd rejects a width of one", {
  expect_error(rollSd(1:10, width = 1), "at least 2")
  expect_equal(rollSd(c(1, 2, 3, 4), width = 2)[2:4], rep(sqrt(0.5), 3))
})

test_that("rolling functions keep ts attributes", {
  x <- ts(as.numeric(1:24), start = c(2020, 1), frequency = 12)
  out <- rollMean(x, width = 3)
  expect_s3_class(out, "ts")
  expect_equal(tsp(out), tsp(x))
})

test_that("naRm controls missing values inside windows", {
  x <- c(1, NA, 3, 4, 5)
  expect_true(is.na(rollMean(x, width = 2)[2]))
  expect_equal(rollMean(x, width = 2, naRm = TRUE)[2], 1)
})

test_that("checkRollArgs validates width and input type", {
  expect_error(rollMean(letters, width = 2), "must be numeric")
  expect_error(rollMean(1:5, width = 0), "single positive number")
  expect_error(rollMean(1:5, width = 6), "exceeds the series length")
})

test_that("rollBySite adds one column per statistic", {
  df <- data.frame(
    site = rep(c("north", "south"), each = 5),
    time = rep(5:1, 2),
    temp = c(10, 11, 12, 13, 14, 20, 21, 22, 23, 24)
  )

  out <- rollBySite(df, "temp", width = 3L, stats = c("mean", "max"))
  expect_named(out, c("site", "time", "temp", "temp_rollMean", "temp_rollMax"))
  # Rows come back sorted by time within each site
  expect_equal(out$time[1:5], 1:5)
  expect_equal(out$temp_rollMax[out$site == "north"], c(NA, NA, 14, 13, 12))
})

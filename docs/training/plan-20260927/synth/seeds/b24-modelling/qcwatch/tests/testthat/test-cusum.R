test_that("cusum.tabular stays at zero for an on-target process", {
  x <- rep(10, 20)
  res <- cusum.tabular(x, target = 10, sigma = 1)
  expect_s3_class(res, "qc_cusum")
  expect_true(all(res$upper == 0))
  expect_true(all(res$lower == 0))
  expect_length(res$signals.upper, 0L)
})

test_that("cusum.tabular detects an upward shift on the upper side", {
  x <- c(rep(0, 10), rep(2, 10))
  res <- cusum.tabular(x, target = 0, sigma = 1, k = 0.5, h = 4)
  expect_gt(length(res$signals.upper), 0L)
  expect_length(res$signals.lower, 0L)
  expect_equal(res$signals.upper[1L], 13L)
})

test_that("cusum.tabular detects a downward shift on the lower side", {
  x <- c(rep(0, 10), rep(-2, 10))
  res <- cusum.tabular(x, target = 0, sigma = 1, k = 0.5, h = 4)
  expect_gt(length(res$signals.lower), 0L)
  expect_length(res$signals.upper, 0L)
  expect_equal(res$signals.lower[1L], 13L)
})

test_that("fast initial response starts both sums at a head start", {
  x <- rep(0, 5)
  res <- cusum.tabular(x, target = 0, sigma = 1, k = 0.5, h = 4, fir = 0.5)
  expect_equal(res$upper[1L], 1.5)
  expect_equal(res$lower[1L], 1.5)
  expect_error(cusum.tabular(x, target = 0, sigma = 1, fir = 1), "fir")
})

test_that("cusum.restart resets the sums after an alarm", {
  x <- c(rep(3, 4), rep(0, 4))
  res <- cusum.restart(x, target = 0, sigma = 1, k = 0.5, h = 4)
  first <- res$signals.upper[1L]
  expect_equal(first, 2L)
  expect_equal(res$upper[first + 1L], 2.5)
})

test_that("cusum.signals estimates the change point", {
  x <- c(rep(0, 10), rep(2, 10))
  res <- cusum.tabular(x, target = 0, sigma = 1, k = 0.5, h = 4)
  sig <- cusum.signals(res)
  expect_equal(nrow(sig), 1L)
  expect_equal(sig$change.point, 11L)
  expect_equal(sig$side, "upper")
  expect_equal(sig$shift, 2, tolerance = 1e-8)
})

test_that("cusum.arl decreases with the size of the shift", {
  arl0 <- cusum.arl(k = 0.5, h = 5, delta = 0)
  arl1 <- cusum.arl(k = 0.5, h = 5, delta = 1)
  arl2 <- cusum.arl(k = 0.5, h = 5, delta = 2)
  expect_gt(arl0, 300)
  expect_lt(arl1, arl0)
  expect_lt(arl2, arl1)
})

test_that("cusum.design recovers the requested in-control ARL", {
  design <- cusum.design(target.arl = 370, k = 0.5)
  expect_equal(design$arl.in.control, 370, tolerance = 1e-3)
  expect_gt(design$h, 4)
  expect_lt(design$h, 6)
  expect_error(cusum.design(target.arl = 1e12, interval = c(0.5, 1)), "widen")
})

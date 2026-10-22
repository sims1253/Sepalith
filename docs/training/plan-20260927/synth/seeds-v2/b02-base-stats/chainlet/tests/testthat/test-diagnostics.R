test_that("autocorrAtLags is near zero for white noise", {
  set.seed(10)
  rho <- autocorrAtLags(rnorm(5000), maxLag = 5)
  expect_length(rho, 5)
  expect_true(all(abs(rho) < 0.05))
})

test_that("autocorrAtLags caps maxLag at n - 1", {
  rho <- autocorrAtLags(c(1, 3, 2, 5), maxLag = 10)
  expect_length(rho, 3)
})

test_that("effectiveSize is close to n for independent draws", {
  set.seed(11)
  ess <- effectiveSize(rnorm(2000))
  expect_gt(ess, 1500)
  expect_lte(ess, 2000)
})

test_that("effectiveSize is much smaller than n for an AR(1) chain", {
  set.seed(12)
  x <- as.numeric(stats::arima.sim(list(ar = 0.9), n = 2000))
  # Theory gives n * (1 - 0.9) / (1 + 0.9), roughly 105.
  expect_lt(effectiveSize(x), 400)
})

test_that("gelmanRubin is close to one for chains from the same target", {
  set.seed(13)
  chains <- replicate(4, rnorm(1000), simplify = FALSE)
  expect_equal(gelmanRubin(chains), 1, tolerance = 0.02)
})

test_that("gelmanRubin is large for chains stuck in different places", {
  set.seed(14)
  chains <- list(rnorm(500, 0), rnorm(500, 5), rnorm(500, -5))
  expect_gt(gelmanRubin(chains), 2)
})

test_that("gelmanRubin rejects a single chain", {
  expect_error(gelmanRubin(list(rnorm(10))), "at least two chains")
})

test_that("burnAndThin keeps the expected rows", {
  chain <- matrix(seq_len(20), ncol = 2)
  out <- burnAndThin(chain, burnIn = 2, thin = 3)
  expect_equal(out[, 1], c(3, 6, 9))

  expect_error(burnAndThin(chain, burnIn = 10), "whole chain")
})

test_that("summarizeChain returns one row per parameter", {
  set.seed(15)
  chain <- cbind(a = rnorm(300), b = rexp(300))
  s <- summarizeChain(chain)
  expect_s3_class(s, "chainletSummary")
  expect_equal(s$param, c("a", "b"))
  expect_output(print(s), "2 parameter")
})

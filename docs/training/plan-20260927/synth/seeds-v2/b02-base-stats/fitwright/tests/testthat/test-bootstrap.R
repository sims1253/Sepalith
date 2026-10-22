test_that("boot_indices returns an R x n matrix of valid rows", {
  set.seed(1)
  idx <- boot_indices(10, 25)
  expect_equal(dim(idx), c(25L, 10L))
  expect_true(all(idx >= 1 & idx <= 10))
})

test_that("boot_indices keeps resamples inside their stratum", {
  set.seed(2)
  strata <- rep(c("a", "b"), times = c(4, 6))
  idx <- boot_indices(10, 50, strata = strata)

  # Columns 1-4 belong to stratum "a", the rest to "b".
  expect_true(all(idx[, 1:4] <= 4))
  expect_true(all(idx[, 5:10] >= 5))
})

test_that("boot_stat names unnamed statistics and stores replicates", {
  set.seed(3)
  res <- boot_stat(rnorm(30), mean, R = 99)
  expect_s3_class(res, "fw_boot")
  expect_equal(names(res$t0), "t1")
  expect_equal(dim(res$t), c(99L, 1L))
})

test_that("percentile interval brackets the original estimate", {
  set.seed(4)
  x <- rnorm(200, mean = 5)
  res <- boot_stat(x, function(d) c(mean = mean(d)), R = 499)
  ci <- boot_ci(res, type = "percentile")
  expect_lt(ci[1, "lower"], res$t0[["mean"]])
  expect_gt(ci[1, "upper"], res$t0[["mean"]])
})

test_that("normal interval brackets the original estimate", {
  set.seed(5)
  x <- rexp(200)
  res <- boot_stat(x, function(d) c(mean = mean(d)), R = 499)
  ci <- boot_ci(res, type = "normal")
  expect_lt(ci[1, "lower"], res$t0[["mean"]])
  expect_gt(ci[1, "upper"], res$t0[["mean"]])
})

test_that("boot_ci rejects objects of the wrong class", {
  expect_error(boot_ci(list(t0 = 1, t = matrix(1))), "fw_boot")
})

test_that("boot_coefs returns one column per coefficient", {
  set.seed(6)
  df <- data.frame(x = runif(40))
  df$y <- 1 + 2 * df$x + rnorm(40, sd = 0.1)
  res <- boot_coefs(y ~ x, df, R = 50)

  expect_equal(colnames(res$t), c("(Intercept)", "x"))
  expect_equal(unname(res$t0[["x"]]), 2, tolerance = 0.2)
})

test_that("summary.fw_boot reports the requested level", {
  set.seed(7)
  res <- boot_stat(rnorm(50), function(d) c(sd = sd(d)), R = 99)
  s <- summary(res, level = 0.9)
  expect_s3_class(s, "summary.fw_boot")
  expect_equal(s$level, 0.9)
  expect_output(print(s), "90% percentile CI")
})

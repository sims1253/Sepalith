items <- data.frame(
  value = c(60, 100, 120, 30),
  weight = c(10L, 20L, 30L, 5L)
)

test_that("knapsack.dp finds the optimal subset", {
  res <- knapsack.dp(items$value, items$weight, capacity = 50L)
  expect_equal(res$value, 220)
  expect_lte(res$weight, 50L)
  expect_equal(which(res$chosen), c(2L, 3L))
})

test_that("knapsack.greedy respects capacity and is within half of optimal", {
  res <- knapsack.greedy(items$value, items$weight, capacity = 50L)
  opt <- knapsack.dp(items$value, items$weight, capacity = 50L)

  expect_lte(res$weight, 50L)
  expect_gte(res$value, opt$value / 2)
  expect_type(res$chosen, "logical")
})

test_that("knapsack.greedy falls back to the best single item", {
  res <- knapsack.greedy(c(1, 1, 10), c(1L, 1L, 10L), capacity = 10L)
  expect_equal(res$value, 10)
  expect_equal(which(res$chosen), 3L)
})

test_that("knapsack solvers handle zero capacity", {
  dp <- knapsack.dp(items$value, items$weight, capacity = 0L)
  greedy <- knapsack.greedy(items$value, items$weight, capacity = 0L)

  expect_equal(dp$value, 0)
  expect_equal(greedy$value, 0)
  expect_false(any(dp$chosen))
  expect_false(any(greedy$chosen))
})

test_that("binpack.first.fit never overfills a bin", {
  size <- c(4, 8, 1, 4, 2, 1, 7, 3)
  bin <- binpack.first.fit(size, capacity = 10)
  loads <- tapply(size, bin, sum)

  expect_length(bin, length(size))
  expect_true(all(loads <= 10))
  expect_gte(max(bin), binpack.lower.bound(size, 10))
})

test_that("binpack.best.fit never overfills a bin", {
  size <- c(4, 8, 1, 4, 2, 1, 7, 3)
  bin <- binpack.best.fit(size, capacity = 10)
  loads <- tapply(size, bin, sum)

  expect_length(bin, length(size))
  expect_true(all(loads <= 10))
  expect_gte(max(bin), binpack.lower.bound(size, 10))
})

test_that("bin packers reject oversized items", {
  expect_error(binpack.first.fit(c(3, 12), capacity = 10), "larger than")
  expect_error(binpack.best.fit(c(3, 12), capacity = 10), "larger than")
})

test_that("binpack.summary reports loads and the gap to the bound", {
  size <- c(6, 6, 5, 3)
  bin <- binpack.first.fit(size, capacity = 10)
  summ <- binpack.summary(size, bin, capacity = 10)

  expect_equal(sum(summ$load), sum(size))
  expect_equal(sum(summ$n.items), length(size))
  # Two items exceed half a bin, so at least two bins are needed
  expect_equal(attr(summ, "lower.bound"), 2L)
  expect_equal(attr(summ, "gap"), nrow(summ) - 2L)
})

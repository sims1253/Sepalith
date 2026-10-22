makeSales <- function() {
  tibble::tibble(
    transactionId = c(1L, 1L, 2L, 3L, 4L, 5L),
    storeId = c("S01", "S01", "S02", "S01", "S02", "S03"),
    category = c("dairy", "bakery", "dairy", "produce", "bakery", "dairy"),
    productId = c("P1", "P2", "P1", "P3", "P2", "P4"),
    saleDate = as.Date(c(
      "2026-03-02", "2026-03-02", "2026-03-04", "2026-03-10", "2026-03-11", "2026-03-17"
    )),
    units = c(2, 1, 3, 4, 2, 1),
    revenue = c(3.0, 2.5, 4.5, 6.0, 5.5, 1.5)
  )
}

test_that("summariseByStore totals revenue, units and transactions", {
  out <- summariseByStore(makeSales())
  expect_equal(out$storeId, c("S01", "S02", "S03"))
  expect_equal(out$revenue, c(11.5, 10.0, 1.5))
  expect_equal(out$transactions, c(2L, 2L, 1L))
  expect_equal(out$avgBasket, out$revenue / out$transactions)
})

test_that("summariseByCategory totals revenue, units and transactions", {
  out <- summariseByCategory(makeSales())
  expect_equal(out$category, c("dairy", "bakery", "produce"))
  expect_equal(out$revenue, c(9.0, 8.0, 6.0))
  expect_equal(out$units, c(6, 3, 4))
})

test_that("summariseByWeek returns weeks in calendar order", {
  out <- summariseByWeek(makeSales())
  expect_equal(out$week, as.Date(c("2026-03-02", "2026-03-09", "2026-03-16")))
  expect_equal(out$revenue, c(10.0, 11.5, 1.5))
})

test_that("summariseSales errors on missing columns", {
  sales <- dplyr::select(makeSales(), -transactionId)
  expect_error(summariseSales(sales, storeId), "missing required columns")
})

test_that("topProducts keeps the n best products per group", {
  out <- topProducts(makeSales(), n = 2L)
  expect_equal(out$productId, c("P2", "P1"))
  expect_equal(out$rank, c(1L, 2L))
  byUnits <- topProducts(makeSales(), n = 1L, by = "units")
  expect_named(byUnits, c("productId", "units", "rank"))
  expect_equal(byUnits$productId, "P1")
})

test_that("bottomProducts keeps the n worst products per group", {
  out <- bottomProducts(makeSales(), n = 2L)
  expect_equal(out$productId, c("P4", "P3"))
  expect_equal(out$rank, c(1L, 2L))
})

test_that("addShare computes shares that sum to one", {
  out <- summariseByStore(makeSales()) |> addShare()
  expect_equal(sum(out$share), 1)
  expect_equal(out$share[[1]], 11.5 / 23)
})

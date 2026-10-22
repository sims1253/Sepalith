fakeOrders <- function() {
  dbplyr::lazy_frame(
    day = as.Date("2024-01-01"),
    region = "EU",
    segment = "retail",
    amount = 10,
    customer_id = 1L,
    con = dbplyr::simulate_postgres()
  )
}

renderSql <- function(lazy) {
  as.character(dbplyr::remote_query(lazy))
}

test_that("filterWindow adds one predicate per bound", {
  sql <- renderSql(filterWindow(fakeOrders(), from = "2024-01-01"))
  expect_match(sql, "day\" >= '2024-01-01'", fixed = TRUE)
  expect_no_match(sql, "<")

  sql <- renderSql(filterWindow(fakeOrders(), from = "2024-01-01", to = "2024-02-01"))
  expect_match(sql, "day\" < '2024-02-01'", fixed = TRUE)
})

test_that("filterWindow without bounds returns the input", {
  lazy <- fakeOrders()
  expect_identical(filterWindow(lazy), lazy)
})

test_that("filterRegions upper-cases and deduplicates codes", {
  sql <- renderSql(filterRegions(fakeOrders(), c("eu", "EU", "us")))
  expect_match(sql, "IN ('EU', 'US')", fixed = TRUE)
  expect_identical(filterRegions(fakeOrders(), NULL), fakeOrders())
})

test_that("filterSegments lower-cases and deduplicates names", {
  sql <- renderSql(filterSegments(fakeOrders(), c("Retail", "retail", "B2B")))
  expect_match(sql, "IN ('retail', 'b2b')", fixed = TRUE)
  expect_identical(filterSegments(fakeOrders(), character()), fakeOrders())
})

test_that("filterBetween validates and applies its bounds", {
  expect_error(filterBetween(fakeOrders(), "amount", 10, 5), "must not exceed")

  # dbplyr renders numeric literals with a trailing .0
  sql <- renderSql(filterBetween(fakeOrders(), "amount", lower = 5, upper = 50))
  expect_match(sql, "amount\" >= 5.0", fixed = TRUE)
  expect_match(sql, "amount\" <= 50.0", fixed = TRUE)
})

test_that("applyFilterSpec rejects unknown filter names", {
  expect_error(applyFilterSpec(fakeOrders(), list(colour = "red")), "Unknown filter")

  sql <- renderSql(applyFilterSpec(fakeOrders(), list(regions = "eu", segments = "retail")))
  expect_match(sql, "region", fixed = TRUE)
  expect_match(sql, "segment", fixed = TRUE)
})

test_that("latestPerGroup uses a ROW_NUMBER window", {
  sql <- renderSql(latestPerGroup(fakeOrders(), "customer_id", "day"))
  expect_match(sql, "ROW_NUMBER() OVER", fixed = TRUE)
  expect_match(sql, "PARTITION BY \"customer_id\"", fixed = TRUE)
  expect_match(sql, "ORDER BY \"day\" DESC", fixed = TRUE)
})

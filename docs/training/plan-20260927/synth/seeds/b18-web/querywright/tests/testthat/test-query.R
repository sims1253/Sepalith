test_that("query_build encodes values and drops NULL and NA", {
  params <- list(q = "red shoes", page = 2, debug = NULL, tag = c("a&b", NA, "c"))
  expect_equal(query_build(params), "q=red%20shoes&page=2&tag=a%26b&tag=c")
  expect_equal(query_build(list(q = "red shoes"), space_as_plus = TRUE), "q=red+shoes")
  expect_equal(query_build(list()), "")
})

test_that("query_build formats logicals and dates", {
  params <- list(active = TRUE, archived = FALSE, from = as.Date("2026-03-01"))
  expect_equal(query_build(params), "active=true&archived=false&from=2026-03-01")
  expect_error(query_build(list(1, 2)), "must be named")
})

test_that("query_parse collects repeated keys and decodes values", {
  params <- query_parse("?tag=a%26b&tag=c&q=red+shoes&flag")
  expect_equal(params$tag, c("a&b", "c"))
  expect_equal(params$q, "red shoes")
  expect_equal(params$flag, "")
  expect_equal(query_parse(""), list())
})

test_that("query_build and query_parse round-trip", {
  params <- list(q = "50% off", sort = "price desc", tag = c("x", "y"))
  expect_equal(query_parse(query_build(params)), params)
})

test_that("url_modify_query adds, replaces and removes parameters", {
  url <- "https://shop.example.org/search?q=boots&page=3#results"
  expect_equal(
    url_modify_query(url, page = 1),
    "https://shop.example.org/search?q=boots&page=1#results"
  )
  expect_equal(
    url_modify_query(url, page = NULL, size = 42),
    "https://shop.example.org/search?q=boots&size=42#results"
  )
})

test_that("url_drop_query removes exact keys and prefixes", {
  url <- "https://example.org/a?id=7&utm_source=mail&utm_medium=x&ref=home"
  expect_equal(url_drop_query(url, c("utm_*", "ref")), "https://example.org/a?id=7")
  expect_equal(url_drop_query("https://example.org/a", "id"), "https://example.org/a")
})

test_that("url_keep_query keeps only the listed keys", {
  url <- "https://example.org/a?id=7&utm_source=mail&utm_medium=x&ref=home"
  expect_equal(url_keep_query(url, "id"), "https://example.org/a?id=7")
  expect_equal(
    url_keep_query(url, "utm_*"),
    "https://example.org/a?utm_source=mail&utm_medium=x"
  )
})

test_that("url_sort_query orders parameters by name", {
  expect_equal(
    url_sort_query("https://example.org/?b=2&a=1&c=3"),
    "https://example.org/?a=1&b=2&c=3"
  )
  expect_equal(url_sort_query("https://example.org/?a=1"), "https://example.org/?a=1")
})

local_clean_cache <- function(env = parent.frame()) {
  cache_clear()
  withr::defer(cache_clear(), envir = env)
  local_tallyhook_options(verbose = FALSE, .local_envir = env)
}

test_that("cache_key() is stable and namespaced", {
  key_a <- cache_key("balance", account = "cash", year = 2025)
  key_b <- cache_key("balance", account = "cash", year = 2025)
  key_c <- cache_key("totals", account = "cash", year = 2025)
  expect_identical(key_a, key_b)
  expect_false(identical(key_a, key_c))
  expect_true(startsWith(key_a, "balance-"))
  expect_error(cache_key(1), class = "tallyhook_error_string")
})

test_that("cache_set() and cache_get() round-trip values", {
  local_clean_cache()
  expect_false(cache_has("ledger-1"))
  expect_null(cache_get("ledger-1"))
  expect_identical(cache_get("ledger-1", default = NA), NA)
  expect_invisible(cache_set("ledger-1", data.frame(x = 1:3)))
  expect_true(cache_has("ledger-1"))
  expect_equal(cache_get("ledger-1"), data.frame(x = 1:3))
})

test_that("expired entries are dropped on access", {
  local_clean_cache()
  cache_set("ledger-short", 1, ttl = 0)
  cache_set("ledger-long", 2, ttl = 3600)
  Sys.sleep(0.01)
  expect_false(cache_has("ledger-short"))
  expect_true(cache_has("ledger-long"))
  expect_error(cache_set("ledger-bad", 3, ttl = -1), class = "tallyhook_error_number")
})

test_that("cache_clear() removes only the requested namespace", {
  local_clean_cache()
  cache_set("balance-a", 1)
  cache_set("balance-b", 2)
  cache_set("totals-a", 3)
  expect_identical(cache_clear("balance"), 2L)
  expect_false(cache_has("balance-a"))
  expect_true(cache_has("totals-a"))
  expect_identical(cache_clear(), 1L)
})

test_that("cache_clear() reports removals when verbose", {
  local_clean_cache()
  local_tallyhook_options(verbose = TRUE)
  cache_set("balance-a", 1)
  cache_set("balance-b", 2)
  expect_snapshot(cache_clear("balance"))
  expect_silent(cache_clear("balance"))
})

test_that("memoise_tally() calls the wrapped function once per input", {
  local_clean_cache()
  calls <- 0L
  count_calls <- function(x) {
    calls <<- calls + 1L
    x * 2
  }
  memo <- memoise_tally(count_calls, "double")
  expect_identical(memo(2), 4)
  expect_identical(memo(2), 4)
  expect_identical(memo(5), 10)
  expect_identical(calls, 2L)
  expect_error(memoise_tally("not a function", "x"), class = "tallyhook_error_function")
})

test_that("memoise_tally() does not cache errors", {
  local_clean_cache()
  attempts <- 0L
  flaky <- function() {
    attempts <<- attempts + 1L
    if (attempts == 1L) stop("first call fails")
    "ok"
  }
  memo <- memoise_tally(flaky, "flaky")
  expect_error(memo(), "first call fails")
  expect_identical(memo(), "ok")
  expect_identical(attempts, 2L)
})

test_that("disk cache round-trips and discards corrupt files", {
  skip_on_cran()
  local_tallyhook_options(cache_dir = withr::local_tempdir())
  expect_null(disk_cache_read("missing"))
  path <- disk_cache_write("report-2025", list(total = 125.5))
  expect_true(file.exists(path))
  expect_identical(disk_cache_read("report-2025"), list(total = 125.5))
  writeLines("not an rds file", path)
  expect_warning(result <- disk_cache_read("report-2025"), "unreadable")
  expect_null(result)
  expect_false(file.exists(path))
})

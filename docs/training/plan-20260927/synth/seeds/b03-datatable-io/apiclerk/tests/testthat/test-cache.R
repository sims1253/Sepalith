local_cache <- function(ttl = 60, env = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = env)
  new_cache(dir, ttl = ttl)
}

test_that("new_cache creates the directory and validates ttl", {
  dir <- file.path(withr::local_tempdir(), "nested", "cache")
  cache <- new_cache(dir, ttl = 10)
  expect_true(dir.exists(dir))
  expect_s3_class(cache, "clerk_cache")
  expect_error(new_cache(dir, ttl = -1))
})

test_that("cache_key ignores query parameter order", {
  k1 <- cache_key("items", list(page = 2, sort = "name"))
  k2 <- cache_key("items", list(sort = "name", page = 2))
  k3 <- cache_key("items", list(sort = "name", page = 3))
  expect_identical(k1, k2)
  expect_false(identical(k1, k3))
  expect_match(k1, "^items-[0-9a-f]{8}$")
})

test_that("cache_set and cache_get round-trip values", {
  cache <- local_cache()
  value <- list(items = list(list(id = 1L, name = "bolt")), next_cursor = "abc")
  expect_invisible(cache_set(cache, "k1", value))
  expect_true(cache_has(cache, "k1"))
  expect_identical(cache_get(cache, "k1"), value)
})

test_that("cache_get returns the default for missing keys", {
  cache <- local_cache()
  expect_false(cache_has(cache, "nope"))
  expect_null(cache_get(cache, "nope"))
  expect_identical(cache_get(cache, "nope", default = list()), list())
})

test_that("entries older than the ttl are treated as missing", {
  cache <- local_cache(ttl = 5)
  cache_set(cache, "old", 42)
  Sys.setFileTime(file.path(cache$dir, "old.rds"), Sys.time() - 60)
  expect_false(cache_has(cache, "old"))
  expect_null(cache_get(cache, "old"))
})

test_that("cache_clear removes only expired entries when asked", {
  cache <- local_cache(ttl = 5)
  cache_set(cache, "old", 1)
  cache_set(cache, "new", 2)
  Sys.setFileTime(file.path(cache$dir, "old.rds"), Sys.time() - 60)
  expect_equal(cache_clear(cache, expired_only = TRUE), 1L)
  expect_true(cache_has(cache, "new"))
  expect_equal(cache_clear(cache), 1L)
  expect_length(list.files(cache$dir), 0L)
})

test_that("cached_get only calls the API on a miss", {
  cache <- local_cache()
  calls <- 0L
  local_mocked_bindings(clerk_get = function(path, query, config) {
    calls <<- calls + 1L
    list(path = path, n = calls)
  })
  cfg <- list(token = "test")
  first <- cached_get("items", list(page = 1), cache = cache, config = cfg)
  second <- cached_get("items", list(page = 1), cache = cache, config = cfg)
  expect_identical(first, second)
  expect_equal(calls, 1L)
  cached_get("items", list(page = 1), cache = cache, refresh = TRUE, config = cfg)
  expect_equal(calls, 2L)
})

test_that("print.clerk_cache reports the number of entries", {
  cache <- local_cache()
  cache_set(cache, "a", letters)
  cache_set(cache, "b", LETTERS)
  expect_output(print(cache), "entries: 2")
  expect_output(print(cache), "ttl:     60 s")
})

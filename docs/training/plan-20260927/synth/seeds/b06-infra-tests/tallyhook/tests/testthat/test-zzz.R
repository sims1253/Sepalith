test_that("default options all carry the tallyhook prefix", {
  defaults <- tallyhook_default_options()
  expect_type(defaults, "list")
  expect_true(all(startsWith(names(defaults), "tallyhook.")))
  expect_named(
    defaults,
    c("tallyhook.verbose", "tallyhook.cache_dir", "tallyhook.max_retries", "tallyhook.currency")
  )
})

test_that(".onLoad() fills unset options but keeps user values", {
  withr::local_options(list(
    tallyhook.verbose = NULL,
    tallyhook.currency = "CHF",
    tallyhook.max_retries = NULL
  ))
  .onLoad(libname = NULL, pkgname = "tallyhook")
  expect_true(getOption("tallyhook.verbose"))
  expect_identical(getOption("tallyhook.currency"), "CHF")
  expect_identical(getOption("tallyhook.max_retries"), 3L)
})

test_that(".onAttach() is silent outside interactive sessions", {
  rlang::local_interactive(FALSE)
  expect_silent(.onAttach(libname = NULL, pkgname = "tallyhook"))
})

test_that("tallyhook_opt() falls back to package defaults then to `default`", {
  withr::local_options(tallyhook.currency = NULL)
  expect_identical(tallyhook_opt("currency"), "EUR")
  expect_null(tallyhook_opt("not_an_option"))
  expect_identical(tallyhook_opt("not_an_option", default = 7), 7)
  expect_error(tallyhook_opt(c("a", "b")), class = "tallyhook_error_string")
})

test_that("local_tallyhook_options() restores values on exit", {
  withr::local_options(tallyhook.currency = "EUR")
  read_inside <- function() {
    local_tallyhook_options(currency = "USD", max_retries = 0L)
    list(tallyhook_opt("currency"), tallyhook_opt("max_retries"))
  }
  expect_identical(read_inside(), list("USD", 0L))
  expect_identical(getOption("tallyhook.currency"), "EUR")
  expect_error(local_tallyhook_options("USD"), "must be named")
})

test_that("option getters validate their values", {
  local_tallyhook_options(verbose = "yes")
  expect_error(tallyhook_is_verbose(), class = "tallyhook_error_flag")
  local_tallyhook_options(verbose = FALSE, max_retries = -2)
  expect_false(tallyhook_is_verbose())
  expect_error(tallyhook_max_retries(), class = "tallyhook_error_count")
  local_tallyhook_options(max_retries = 5)
  expect_identical(tallyhook_max_retries(), 5L)
})

test_that("tallyhook_cache_dir() creates the directory on request", {
  root <- withr::local_tempdir()
  target <- file.path(root, "nested", "cache")
  local_tallyhook_options(cache_dir = target)
  expect_identical(tallyhook_cache_dir(create = FALSE), target)
  expect_false(dir.exists(target))
  expect_identical(tallyhook_cache_dir(), target)
  expect_true(dir.exists(target))
})

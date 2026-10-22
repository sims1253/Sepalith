test_that("is_quiet() prefers the option over the environment variable", {
  withr::local_options(burrowlog.quiet = NULL)
  withr::local_envvar(BURROWLOG_QUIET = "true")
  expect_true(is_quiet())
  withr::local_envvar(BURROWLOG_QUIET = "no")
  expect_false(is_quiet())
  withr::local_options(burrowlog.quiet = TRUE)
  expect_true(is_quiet())
})

test_that("message helpers are classed and respect quiet mode", {
  withr::local_options(burrowlog.quiet = FALSE)
  expect_message(burrow_inform("Loaded {3} files."), class = "burrowlog_message")
  expect_message(burrow_success("Done."), class = "burrowlog_message")
  expect_message(burrow_warn("Skipped {.file a.csv}."), class = "burrowlog_message")
  withr::local_options(burrowlog.quiet = TRUE)
  expect_silent(burrow_inform("Loaded {3} files."))
  expect_silent(burrow_success("Done."))
  expect_silent(burrow_warn("Skipped {.file a.csv}."))
})

test_that("message helpers interpolate in the caller's frame", {
  withr::local_options(burrowlog.quiet = FALSE)
  n_rows <- 42
  expect_snapshot({
    burrow_inform("Read {n_rows} rows.")
    burrow_success("Wrote {n_rows} rows.")
    burrow_warn("Dropped {n_rows} rows.")
  })
})

test_that("format_summary_bullets() aligns labels and escapes braces", {
  out <- format_summary_bullets(c(rows = 1200, dropped = 3))
  expect_named(out, c("*", "*"))
  expect_identical(unname(out), c("rows    : 1,200", "dropped : 3"))
  out <- format_summary_bullets(c(pattern = "{x}"), bullet = "i")
  expect_identical(unname(out), "pattern : {{x}}")
  expect_identical(format_summary_bullets(numeric()), character())
  expect_error(format_summary_bullets(c(1, 2)), class = "burrowlog_error_unnamed")
  expect_error(format_summary_bullets(c(a = 1), bullet = "?"))
})

test_that("burrow_map() returns a named list like lapply()", {
  withr::local_options(burrowlog.quiet = TRUE)
  x <- c(a = 1, b = 2, c = 3)
  expect_identical(burrow_map(x, function(v) v * 10), lapply(x, function(v) v * 10))
  expect_identical(burrow_map(1:2, ~ .x + 1L), list(2L, 3L))
  expect_identical(burrow_map(list(), identity), list())
  expect_identical(burrow_map(1:2, rep, times = 2), list(c(1L, 1L), c(2L, 2L)))
})

test_that("burrow_map() reports the failing index", {
  withr::local_options(burrowlog.quiet = TRUE)
  fail_on_two <- function(v) if (v == 2) stop("bad value") else v
  err <- expect_error(burrow_map(1:3, fail_on_two), class = "burrowlog_error_map")
  expect_identical(err$index, 2L)
  expect_match(conditionMessage(err$parent), "bad value")
})

test_that("burrow_map() shows progress when not quiet", {
  skip_if_not_installed("cli", minimum_version = "3.6.0")
  withr::local_options(
    burrowlog.quiet = FALSE,
    cli.progress_show_after = 0,
    cli.dynamic = FALSE
  )
  expect_no_error(burrow_map(1:3, identity, label = "Mapping"))
})

test_that("with_step() returns the value of its code", {
  withr::local_options(burrowlog.quiet = TRUE)
  expect_identical(with_step("Counting", 1 + 1), 2)
  withr::local_options(burrowlog.quiet = FALSE)
  expect_message(value <- with_step("Counting", nrow(mtcars)), "took")
  expect_identical(value, 32L)
})

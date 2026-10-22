test_that("local_quiet() sets and restores the quiet option", {
  withr::local_options(burrowlog.quiet = FALSE)
  inner <- function() {
    local_quiet()
    is_quiet()
  }
  expect_true(inner())
  expect_false(is_quiet())
  expect_error(local_quiet(NA))
})

test_that("with_quiet() mirrors local_quiet() for a block of code", {
  withr::local_options(burrowlog.quiet = FALSE)
  expect_silent(with_quiet(code = burrow_inform("hidden")))
  expect_message(with_quiet(FALSE, burrow_inform("shown")), "shown")
  expect_identical(with_quiet(code = 1 + 1), 2)
  expect_false(getOption("burrowlog.quiet"))
})

test_that("local_burrow_seed() gives reproducible draws", {
  first <- with_burrow_seed(2025, stats::runif(3))
  second <- with_burrow_seed(2025, stats::runif(3))
  expect_identical(first, second)
  expect_false(identical(first, with_burrow_seed(2026, stats::runif(3))))
})

test_that("local_burrow_seed() leaves the caller's RNG state untouched", {
  set.seed(1)
  before <- .Random.seed
  with_burrow_seed(99, stats::rnorm(10))
  expect_identical(.Random.seed, before)
  expect_error(local_burrow_seed("99"))
})

test_that("local_burrow_project() writes files and switches directory", {
  old_wd <- getwd()
  inner <- function() {
    root <- local_burrow_project(list(
      "DESCRIPTION" = c("Package: demo", "Version: 0.1.0"),
      "R/hello.R" = "hello <- function() 'hi'"
    ))
    list(
      root = root,
      wd = getwd(),
      files = sort(list.files(recursive = TRUE)),
      desc = readLines("DESCRIPTION")
    )
  }
  res <- inner()
  expect_identical(normalizePath(res$wd), normalizePath(res$root))
  expect_identical(res$files, c("DESCRIPTION", "R/hello.R"))
  expect_identical(res$desc[1], "Package: demo")
  expect_identical(getwd(), old_wd)
  expect_false(dir.exists(res$root))
})

test_that("with_burrow_project() validates file names", {
  expect_error(
    with_burrow_project(list("a", "b"), NULL),
    class = "burrowlog_error_files"
  )
  expect_identical(with_burrow_project(code = length(list.files())), 0L)
})

test_that("log_line() writes to the scoped log file", {
  path <- local_log_file()
  expect_identical(getOption("burrowlog.log_file"), path)
  log_line("info", "started")
  log_line("error", "failed to read input")
  lines <- readLines(path)
  expect_length(lines, 2L)
  expect_match(lines[1], "\\[INFO\\] started$")
  expect_match(lines[2], "\\[ERROR\\] failed to read input$")
})

test_that("log_line() is a no-op without a log file", {
  withr::local_options(burrowlog.log_file = NULL)
  expect_null(log_line("warn", "nobody hears this"))
  expect_error(log_line("debug", "x"))
})

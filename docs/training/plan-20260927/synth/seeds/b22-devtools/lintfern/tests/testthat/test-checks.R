tokensFor <- function(code) {
  path <- withr::local_tempfile(fileext = ".R", .local_envir = parent.frame())
  writeLines(code, path)
  readSourceTokens(path)
}

test_that("findFunctionDefs() names assigned functions and skips anonymous ones", {
  pd <- tokensFor(c(
    "f <- function(x) x + 1",
    "g = function(a, b) {",
    "  lapply(a, function(z) z * b)",
    "}"
  ))
  defs <- findFunctionDefs(pd)
  expect_equal(nrow(defs), 3L)
  expect_setequal(defs$name[!is.na(defs$name)], c("f", "g"))
  expect_equal(sum(is.na(defs$name)), 1L)
})

test_that("checkUnusedArguments() reports only arguments missing from the body", {
  pd <- tokensFor(c(
    "scale_by <- function(x, factor, verbose = FALSE) {",
    "  x * factor",
    "}"
  ))
  hits <- checkUnusedArguments(pd)
  expect_equal(nrow(hits), 1L)
  expect_match(hits$message, "'verbose' of scale_by")
  expect_equal(hits$line, 1L)
})

test_that("checkUnusedArguments() ignores dots", {
  pd <- tokensFor("wrap <- function(x, ...) x")
  expect_equal(nrow(checkUnusedArguments(pd)), 0L)
})

test_that("checkLongFunctions() respects the line limit", {
  body <- sprintf("  y%d <- %d", 1:12, 1:12)
  pd <- tokensFor(c("big <- function() {", body, "}"))
  expect_equal(nrow(checkLongFunctions(pd, maxLines = 20L)), 0L)
  hits <- checkLongFunctions(pd, maxLines = 5L)
  expect_equal(nrow(hits), 1L)
  expect_match(hits$message, "'big' has 14 lines")
})

test_that("checkTodoComments() extracts the tag and note", {
  pd <- tokensFor(c(
    "# TODO: handle NA input",
    "x <- 1 # fixme",
    "# an ordinary comment"
  ))
  hits <- checkTodoComments(pd)
  expect_equal(nrow(hits), 2L)
  expect_equal(hits$message, c("TODO: handle NA input", "Bare FIXME comment."))
})

test_that("checkLineLength() flags only lines over the limit", {
  lines <- c("short", strrep("x", 81), strrep("y", 80))
  hits <- checkLineLength(lines, maxWidth = 80L)
  expect_equal(hits$line, 2L)
  expect_equal(hits$column, 81L)
})

test_that("checkTrailingWhitespace() points just past the last character", {
  hits <- checkTrailingWhitespace(c("a <- 1  ", "b <- 2", "c <- 3 "))
  expect_equal(hits$line, c(1L, 3L))
  expect_equal(hits$column, c(7L, 7L))
})

test_that("lintFile() combines rules into a sorted lintResult", {
  path <- withr::local_tempfile(fileext = ".R")
  writeLines(c("# TODO: remove", "f <- function(x, unused) x"), path)
  res <- lintFile(path, rules = list(ruleTodoComments(), ruleUnusedArguments()))
  expect_s3_class(res, "lintResult")
  expect_equal(res$rule, c("todo_comments", "unused_arguments"))
  expect_equal(res$line, c(1L, 2L))
  expect_output(print(res), "2 lint\\(s\\) in 1 file\\(s\\)")
})

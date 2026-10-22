test_that("check_string() accepts a single string and returns it invisibly", {
  expect_invisible(check_string("ledger"))
  expect_identical(check_string("ledger"), "ledger")
  expect_null(check_string(NULL, allow_null = TRUE))
})

test_that("check_string() rejects non-strings with a classed error", {
  expect_error(check_string(1), class = "tallyhook_error_string")
  expect_error(check_string(c("a", "b")), class = "tallyhook_error_string")
  expect_error(check_string(NA_character_), class = "tallyhook_error_string")
  expect_error(check_string(NULL), class = "tallyhook_error_arg")
})

test_that("check_flag() mirrors check_string() for logicals", {
  expect_identical(check_flag(TRUE), TRUE)
  expect_null(check_flag(NULL, allow_null = TRUE))
  expect_error(check_flag("yes"), class = "tallyhook_error_flag")
  expect_error(check_flag(NA), class = "tallyhook_error_flag")
  expect_error(check_flag(c(TRUE, FALSE)), class = "tallyhook_error_flag")
})

test_that("check_count() coerces whole doubles and rejects the rest", {
  expect_identical(check_count(3), 3L)
  expect_identical(check_count(0), 0L)
  expect_error(check_count(0, allow_zero = FALSE), class = "tallyhook_error_count")
  expect_error(check_count(-1), class = "tallyhook_error_count")
  expect_error(check_count(2.5), class = "tallyhook_error_count")
  expect_error(check_count("3"), class = "tallyhook_error_count")
})

test_that("check_number_between() enforces inclusive bounds", {
  expect_identical(check_number_between(0.5, min = 0, max = 1), 0.5)
  expect_identical(check_number_between(1, min = 0, max = 1), 1)
  expect_error(check_number_between(1.1, min = 0, max = 1), class = "tallyhook_error_number")
  expect_error(check_number_between(Inf), class = "tallyhook_error_number")
  expect_error(check_number_between(NA_real_), class = "tallyhook_error_number")
})

test_that("check_choice() never partially matches", {
  choices <- c("daily", "weekly", "monthly")
  expect_identical(check_choice("weekly", choices), "weekly")
  expect_error(check_choice("week", choices), class = "tallyhook_error_choice")
  expect_error(check_choice(1, choices), class = "tallyhook_error_string")
})

test_that("check_has_columns() reports every missing column", {
  entries <- data.frame(id = 1:2, amount = c(10, 20))
  expect_identical(check_has_columns(entries, c("id", "amount")), entries)
  err <- expect_error(
    check_has_columns(entries, c("id", "posted_at", "account")),
    class = "tallyhook_error_columns"
  )
  expect_identical(err$missing, c("posted_at", "account"))
  expect_error(check_has_columns(list(id = 1), "id"), class = "tallyhook_error_columns")
})

test_that("checkers report the caller's argument name and call", {
  post_entry <- function(account, amount) {
    check_string(account)
    check_number_between(amount, min = 0)
    invisible(TRUE)
  }
  err <- expect_error(post_entry(42, 10), class = "tallyhook_error_arg")
  expect_identical(err$arg, "account")
  expect_snapshot(post_entry("cash", -5), error = TRUE)
  expect_snapshot(post_entry(c("cash", "bank"), 5), error = TRUE)
})

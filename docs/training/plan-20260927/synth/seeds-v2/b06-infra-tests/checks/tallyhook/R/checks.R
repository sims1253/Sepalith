# skipped: needs cli and rlang (abort_arg and every error path of the
#   check_* functions); only the passing paths are exercised here
stopifnot(identical(check_string("ledger"), "ledger"))
stopifnot(is.null(check_string(NULL, allow_null = TRUE)))
stopifnot(identical(check_flag(FALSE), FALSE), is.null(check_flag(NULL, allow_null = TRUE)))

stopifnot(identical(check_count(3), 3L), identical(check_count(0), 0L))
stopifnot(identical(check_count(2L, allow_zero = FALSE), 2L))

stopifnot(identical(check_number_between(1, min = 0, max = 1), 1))
stopifnot(identical(check_number_between(-5), -5))
stopifnot(identical(check_choice("weekly", c("daily", "weekly")), "weekly"))

entries <- data.frame(id = 1:2, amount = c(10, 20))
stopifnot(identical(check_has_columns(entries, c("id", "amount")), entries))
stopifnot(identical(check_has_columns(entries, character()), entries))

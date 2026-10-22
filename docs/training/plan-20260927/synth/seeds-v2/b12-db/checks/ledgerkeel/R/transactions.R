# skipped: needs DBI or glue (with_savepoint, with_retry, account_balance,
# post_debit, post_credit, unbalanced_entries); transfer only on its early exits

# is_retryable
stopifnot(is_retryable(simpleError("ERROR: could not serialize access")))
stopifnot(is_retryable(simpleError("SQLSTATE 40P01")))
stopifnot(!is_retryable(simpleError("relation does not exist")))

# transfer validates before opening a transaction
stopifnot(inherits(try(transfer(NULL, "a", "a", 5), silent = TRUE), "try-error"))
err <- try(transfer(NULL, "a", "b", NA_real_), silent = TRUE)
stopifnot(grepl("single positive number", err))
stopifnot(grepl("single positive number", try(transfer(NULL, "a", "b", -1), silent = TRUE)))

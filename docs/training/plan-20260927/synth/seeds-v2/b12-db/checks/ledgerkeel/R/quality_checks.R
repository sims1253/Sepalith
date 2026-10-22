# skipped: needs glue and DBI (check.not.null, check.unique, check.accepted.values,
# check.in.range, check.relationship); run.checks only on its error paths

# `%||%`
stopifnot(identical(NULL %||% "x", "x"), identical("y" %||% "x", "y"))

# check.result
r <- check.result("unique", "orders", "id", 2L)
stopifnot(nrow(r) == 1, identical(r$failures, 2), identical(r$passed, FALSE))
stopifnot(check.result("not_null", "t", NA, 0)$passed)
stopifnot(is.na(check.result("x", "t", NA, NA)$passed))

# run.checks
empty <- run.checks(NULL, list())
stopifnot(nrow(empty) == 0, identical(names(empty), names(r)))
stopifnot(inherits(try(run.checks(NULL, list(list(type = "bogus"))), silent = TRUE), "try-error"))
spec <- list(list(type = "in_range", table = "t", column = "x", lower = 5, upper = 1))
got <- suppressWarnings(run.checks(NULL, spec))
stopifnot(nrow(got) == 1, identical(got$passed, FALSE), is.na(got$failures))
stopifnot(inherits(try(suppressWarnings(run.checks(NULL, spec, stop.on.fail = TRUE)),
  silent = TRUE), "try-error"))

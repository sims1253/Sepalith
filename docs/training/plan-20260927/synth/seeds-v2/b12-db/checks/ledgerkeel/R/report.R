# skipped: needs dplyr or DBI (summary.check_report, write_check_history,
# read_check_history, flaky_checks)
res <- data.frame(
  check = c("not_null", "unique", "in_range"),
  table = c("orders", "orders", "items"),
  column = c("id", NA, "qty"),
  failures = c(0, 3, NA),
  passed = c(TRUE, FALSE, NA),
  stringsAsFactors = FALSE
)

# new_check_report
rep <- new_check_report(res, "prod")
stopifnot(inherits(rep, "check_report"), identical(rep$database, "prod"))
stopifnot(inherits(try(new_check_report(res[1:2], "x"), silent = TRUE), "try-error"))

# format.check_report
txt <- format(rep)
stopifnot(length(txt) == 3, startsWith(txt[1], "PASS"))
stopifnot(grepl("orders (3 rows)", txt[2], fixed = TRUE))
stopifnot(startsWith(txt[3], "FAIL"), grepl("items.qty (error)", txt[3], fixed = TRUE))

# print.check_report
out <- capture.output(ret <- print(rep))
stopifnot(startsWith(out[1], "<check_report: prod, "), length(out) == 4, identical(ret, rep))

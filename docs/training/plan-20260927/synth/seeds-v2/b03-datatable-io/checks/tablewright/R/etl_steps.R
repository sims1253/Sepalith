# skipped: check_unique_key, check_no_missing, step_rename, step_cast, step_derive,
#   step_filter, log_step, run_steps (need data.table)

df <- data.frame(a = 1, b = 2)
stopifnot(identical(check_has_cols(df, c("a", "b")), df))
msg <- tryCatch(check_has_cols(df, c("a", "x", "y"), "load"), error = conditionMessage)
stopifnot(identical(msg, "[load] missing column(s): x, y"))
stopifnot(grepl("^\\[step\\]", tryCatch(check_has_cols(df, "z"), error = conditionMessage)))

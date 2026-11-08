stopifnot(is.null(problem("a", "missing", integer())))
p <- problem("a", "missing", 1:7)
stopifnot(identical(p$n, 7L), identical(p$rows, "1,2,3,4,5"))

ci <- check_ids(c("a", "a", NA, "b c", ""), "id")
stopifnot(identical(ci$rule, c("missing", "duplicate", "whitespace")))
stopifnot(identical(ci$rows, c("3,5", "2", "4")), nrow(check_ids(c("x", "y"), "id")) == 0L)

ca <- check_amounts(c("1", "x", NA, "-2"), "amt")
stopifnot(identical(ca$rule, c("missing", "not_numeric", "negative")))
stopifnot(identical(ca$rows, c("3", "2", "4")))
stopifnot(nrow(check_amounts(c(3, 5), "amt")) == 0L)

cd <- check_dates(c("2024-01-01", "soon", NA, "2030-01-01"), "d", max_date = "2025-01-01")
stopifnot(identical(cd$rule, c("missing", "not_date", "future")), identical(cd$n, c(1L, 1L, 1L)))
stopifnot(identical(cd$rows[3], "4"))

cl <- check_levels(c("open", "Open", "zz", NA), "s", c("open", "closed"))
stopifnot(identical(cl$rule, c("missing", "unknown_level", "case_mismatch")))
stopifnot(identical(cl$rows, c("4", "2,3", "2")))

df <- data.frame(id = c("a", "a"), amt = c(1, -1), d = c("2024-01-01", "2024-02-01"),
                 s = c("open", "x"), stringsAsFactors = FALSE)
v <- validate_table(df, ids = "id", amounts = "amt", dates = "d",
                    levels = list(s = c("open", "closed")), max_date = "2025-01-01")
stopifnot(inherits(v, "rake_problems"), identical(v$column, c("id", "amt", "s")))
stopifnot(identical(v$rule, c("duplicate", "negative", "unknown_level")))
stopifnot(inherits(try(validate_table(df, ids = "zz"), silent = TRUE), "try-error"))
stopifnot(identical(format(v)[1], "id: duplicate in 1 row(s) [2]"))
stopifnot(identical(format(validate_table(df, dates = "d")), "No problems found."))

stopifnot(identical(check_ids(c("a1", "b#"), "id")$rule, "bad_chars"))
stopifnot(identical(check_amounts(c(0, 1), "amt")$rule, "zero"))
stopifnot(identical(check_dates("1999-12-31", "d")$rule, "past"))
stopifnot(identical(check_dates("1999-12-31", "d", min_date = "1990-01-01")$n, integer()))
cp <- check_levels(c("open ", "open"), "s", "open")
stopifnot(identical(cp$rule, c("unknown_level", "padded")))
stopifnot(nrow(check_ids(c("a", NA), "id", allow_missing = TRUE)) == 0L)
stopifnot(nrow(check_amounts(c(1, NA), "amt", allow_missing = TRUE)) == 0L)
stopifnot(nrow(check_dates(c("2024-01-01", NA), "d", allow_missing = TRUE)) == 0L)
stopifnot(identical(check_levels(c("x", NA), "s", "x", allow_missing = FALSE)$rule, "missing"))
stopifnot(nrow(check_levels(c("x", NA), "s", "x", allow_missing = TRUE)) == 0L)

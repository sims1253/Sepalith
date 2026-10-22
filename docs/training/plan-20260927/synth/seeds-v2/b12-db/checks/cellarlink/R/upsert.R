# skipped: needs DBI and glue (write_staging, upsert_on_conflict, upsert_replace,
# insert_new); upsert_batched is only exercised on its early exits
df <- data.frame(id = c(1, 2, 3), grp = c("a", "a", "b"), v = 1:3)

# check_keys
stopifnot(identical(check_keys(df, "id"), df), identical(check_keys(df, c("grp", "v")), df))
stopifnot(inherits(try(check_keys(df, "grp"), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_keys(df, "nope"), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_keys(df, character()), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_keys(data.frame(id = c(1, NA)), "id"), silent = TRUE), "try-error"))

# upsert_batched: empty input returns before touching the connection
stopifnot(identical(upsert_batched(NULL, "t", df[0, ], "id"), integer()))
stopifnot(inherits(try(upsert_batched(NULL, "t", df, "id", batch_size = 0), silent = TRUE),
  "try-error"))
dup <- data.frame(id = c(1, 1), v = 1:2)
stopifnot(inherits(try(upsert_batched(NULL, "t", dup, "id", batch_size = 1), silent = TRUE),
  "try-error"))

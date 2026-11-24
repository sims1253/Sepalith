p <- column.profile(c(3, 1, NA, 3))
stopifnot(p$n == 4L, p$n_missing == 1L, p$n_distinct == 2L)
stopifnot(p$min == 1, p$max == 3, abs(p$mean - 7 / 3) < 1e-12)
q <- column.profile(c(TRUE, FALSE), c("n", "min", "mean"))
stopifnot(identical(names(q), c("n", "min", "mean")), is.na(q$min), is.na(q$mean))
stopifnot(column.profile(c("b", "a"), "max")$max == "b")
stopifnot(is.na(column.profile(c(NA_real_, NA), "min")$min))

df <- data.frame(
  id = c(2L, 5L), score = c(1.5, NA), ok = c(TRUE, TRUE),
  day = as.Date(c("2024-01-02", "2024-03-04")),
  at = as.POSIXct(c("2024-01-02 10:00:00", NA), tz = "UTC"),
  name = c("ab", "abcd")
)
pr <- result.profile(df)
stopifnot(identical(pr$kind, c("integer", "numeric", "logical", "date", "timestamp", "character")))
stopifnot(pr$low[1] == "2", pr$high[1] == "5", pr$low[2] == "1.5", pr$n_missing[2] == 1L)
stopifnot(pr$low[3] == "0", pr$high[3] == "2", pr$high[4] == "2024-03-04")
stopifnot(pr$low[5] == "2024-01-02 10:00:00", pr$low[6] == "2", pr$high[6] == "4")
stopifnot(is.na(result.profile(data.frame(x = NA_integer_))$low))
stopifnot(inherits(try(result.profile(1:3), silent = TRUE), "try-error"))

old <- data.frame(id = 1:3, v = c("a", "b", "c"))
new <- data.frame(v = c("b", "x", "d"), id = c(2L, 3L, 4L))
cmp <- compare.result.sets(old, new, "id")
stopifnot(cmp$removed$id == 1L, cmp$added$id == 4L, cmp$changed$id == 3L)
stopifnot(cmp$changed$v == "x", identical(names(cmp$added), c("id", "v")))
stopifnot(inherits(try(compare.result.sets(old, new[1], "id"), silent = TRUE), "try-error"))
stopifnot(inherits(try(compare.result.sets(old, new, "zz"), silent = TRUE), "try-error"))

s <- result.set.schema(df)
stopifnot(identical(s$sql_type, c("INTEGER", "DOUBLE PRECISION", "BOOLEAN", "DATE",
                                  "TIMESTAMPTZ", "TEXT")))
stopifnot(identical(s$r_class[4:5], c("Date", "POSIXct")))
sm <- result.set.schema(df, "mysql")
stopifnot(sm$sql_type[1] == "INT", sm$sql_type[3] == "TINYINT(1)", sm$sql_type[5] == "DATETIME")
stopifnot(result.set.schema(df, "sqlite")$sql_type[2] == "REAL")

k <- summarize.kinds(df)
stopifnot(nrow(k) == 6L, all(k$columns == 1L), k$missing[k$kind == "numeric"] == 1L)
k2 <- summarize.kinds(data.frame(a = 1:2, b = c(NA, 3L), c = "x"))
stopifnot(identical(k2$kind, c("integer", "character")), k2$columns[1] == 2L, k2$missing[1] == 1L)

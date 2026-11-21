# skipped: needs dplyr/purrr (group.counts.table, group.stats.table, group.change.table,
# group.top.table, group.tables)

stopifnot(identical(table.caption("counts", "region", "sales"), "Rows per region."))
stopifnot(identical(table.caption("stats", "region", "sales"), "Summary of sales per region."))
stopifnot(identical(table.caption("change", "r", "s"), "Change in s per r between periods."))
stopifnot(identical(table.caption("top", "r", "s", n = 5L), "Top 5 rows by s per r."))
stopifnot(inherits(try(table.caption("other", "r", "s"), silent = TRUE), "try-error"))

tb = data.frame(group = c("b", "a", "b"), n = 1:3, mean = c(1, 2, 3))
fr = per.group.frames(tb)
stopifnot(identical(names(fr), c("b", "a")), identical(names(fr$b), c("n", "mean")))
stopifnot(identical(fr$b$n, c(1L, 3L)), identical(rownames(fr$b), c("1", "2")), nrow(fr$a) == 1L)

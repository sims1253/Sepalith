stopifnot(identical(codebook.entry(c(3L, NA, 1L), "n")$summary, "1 to 3"))
stopifnot(identical(codebook.entry(c(3L, NA, 1L), "n")$missing, 1L))
stopifnot(identical(codebook.entry(c(0.5, 2), "w")$summary, "0.5 to 2"))
stopifnot(identical(codebook.entry(as.Date(c("2026-01-02", "2026-01-01")), "d")$summary,
  "2026-01-01 to 2026-01-02"))
stopifnot(identical(codebook.entry(c("a", "b", "a", NA), "s")$summary, "2 distinct values"))
stopifnot(identical(codebook.entry(c(TRUE, NA, FALSE, TRUE), "l")$summary, "2 TRUE, 1 FALSE"))
stopifnot(identical(codebook.entry(c(NA, NA), "z")$summary, "all missing"))
msg <- tryCatch(codebook.entry(factor("a"), "f"), error = conditionMessage)
stopifnot(identical(msg, "Column `f` has unsupported type `factor`."))

data <- data.frame(id = 1:3, w = c(1.5, NA, 2), name = c("x", "y", "x"),
  ok = c(TRUE, FALSE, NA), stringsAsFactors = FALSE)
cb <- codebook.table(data, "demo")
stopifnot(identical(cb$name, names(data)), identical(cb$table, rep("demo", 4)))
stopifnot(identical(cb$type, c("integer", "numeric", "character", "logical")))
stopifnot(identical(cb$missing, c(0L, 1L, 0L, 1L)))

for (fmt in c("dcf", "csv")) {
  path <- tempfile(fileext = paste0(".", fmt))
  stopifnot(identical(write.codebook(cb, path, format = fmt), path))
  stopifnot(identical(read.codebook(path, format = fmt), cb))
  unlink(path)
}
path <- tempfile(fileext = ".txt")
write.codebook(cb, path, format = "txt")
txt <- readLines(path)
stopifnot(length(txt) == 5, identical(txt[1], "name  type       missing  summary"))
stopifnot(identical(txt[2], "id    integer          0  1 to 3"))
unlink(path)

res <- codebook.check(cb, data)
stopifnot(length(res$missing) == 0, length(res$extra) == 0, length(res$retyped) == 0)
changed <- data[c("id", "w", "ok")]
changed$id <- as.numeric(changed$id)
changed$new <- 1
res <- codebook.check(cb, changed)
stopifnot(identical(res$missing, "name"),
  identical(res$extra, "new"), identical(res$retyped, "id"))
stopifnot(length(res$more.missing) == 0)
changed$w <- NA_real_
stopifnot(identical(codebook.check(cb, changed)$more.missing, "w"))
path <- tempfile(fileext = ".csv")
utils::write.csv(cb[1:3], path, row.names = FALSE)
msg <- tryCatch(read.codebook(path, "csv"), error = conditionMessage)
stopifnot(identical(msg, sprintf("Codebook `%s` is missing field(s): missing, summary", path)))
unlink(path)

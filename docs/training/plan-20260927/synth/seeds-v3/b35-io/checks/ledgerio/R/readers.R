fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")
raw <- data.frame(entry_id = c("2", "1"), posted = c("2026-01-05", "2026-01-02"),
  account = c("cash", "bank"), amount = c("10.5", "-3"), currency = c("eur", "EUR"),
  memo = c("", "fee"), extra = c("x", "y"), stringsAsFactors = FALSE)
out <- coerce_ledger_columns(raw, "mem")
stopifnot(identical(names(out), fields), is.integer(out$entry_id))
stopifnot(inherits(out$posted, "Date"), identical(out$amount, c(10.5, -3)))
stopifnot(identical(out$memo, c("", "fee")), identical(out$currency, c("EUR", "EUR")))
for (col in fields) {
  bad <- raw
  bad[[col]][1] <- if (col %in% c("account", "currency", "memo")) NA else "zz"
  msg <- tryCatch(coerce_ledger_columns(bad, "f.csv"), error = conditionMessage)
  stopifnot(identical(msg, sprintf("`f.csv`: column `%s` has invalid values.", col)))
}
msg <- tryCatch(coerce_ledger_columns(raw[c("entry_id", "memo")], "g"), error = conditionMessage)
stopifnot(identical(msg, "`g` is missing column(s): posted, account, amount, currency"))

dir <- tempfile("ledger")
dir.create(dir)
csv <- file.path(dir, "a.csv")
utils::write.csv(raw[fields], csv, row.names = FALSE)
tsv <- file.path(dir, "b.tsv")
utils::write.table(raw[fields], tsv, sep = "\t", row.names = FALSE, quote = FALSE)
rds <- file.path(dir, "c.rds")
saveRDS(out, rds)
dcf <- file.path(dir, "d.dcf")
write.dcf(raw[fields], dcf)

a <- read_ledger_csv(csv)
stopifnot(identical(attr(a, "format"), "csv"), identical(attr(a, "source"), csv))
stopifnot(identical(a$amount, c(-3, 10.5)), identical(a$memo, c("fee", "")))
stopifnot(identical(read_ledger_csv(csv, sort = FALSE)$entry_id, c(2L, 1L)))
b <- read_ledger_tsv(tsv)
stopifnot(identical(attr(b, "format"), "tsv"), identical(b$entry_id, c(1L, 2L)))
c1 <- read_ledger_rds(rds, sort = FALSE)
stopifnot(identical(attr(c1, "format"), "rds"), identical(c1$posted, out$posted))
bad_rds <- file.path(dir, "bad.rds")
saveRDS(1:3, bad_rds)
msg <- tryCatch(read_ledger_rds(bad_rds), error = conditionMessage)
stopifnot(identical(msg, sprintf("`%s` does not contain a data frame.", bad_rds)))
unlink(bad_rds)
d <- read_ledger_dcf(dcf)
stopifnot(identical(attr(d, "format"), "dcf"), identical(d$account, c("bank", "cash")))
msg <- tryCatch(read_ledger_csv(file.path(dir, "none.csv")), error = conditionMessage)
stopifnot(grepl("does not exist", msg))
empty <- tempfile(fileext = ".tsv")
file.create(empty)
msg <- tryCatch(read_ledger_tsv(empty), error = conditionMessage)
stopifnot(identical(msg, sprintf("Ledger file `%s` is empty.", empty)))
unlink(empty)

for (p in c(csv, tsv, rds, dcf)) {
  stopifnot(identical(attr(read_ledger(p), "format"), tools::file_ext(p)))
}
stopifnot(identical(attr(read_ledger(csv, format = "csv"), "format"), "csv"))
msg <- tryCatch(read_ledger(file.path(dir, "x.txt")), error = conditionMessage)
stopifnot(grepl("Cannot guess", msg))

all <- read_ledger_dir(dir)
stopifnot(nrow(all) == 8, identical(all$file[1:4], c("a.csv", "b.tsv", "c.rds", "d.dcf")))
stopifnot(!is.unsorted(all$posted))
files <- attr(all, "files")
stopifnot(identical(files$format, c("csv", "tsv", "rds", "dcf")),
  identical(files$rows, rep(2L, 4)))
msg <- tryCatch(read_ledger_dir(tempdir(), pattern = "^nothing_here$"), error = conditionMessage)
stopifnot(grepl("No ledger files", msg))
unlink(dir, recursive = TRUE)

ledger <- data.frame(entry_id = c(1L, 2L, 3L),
  posted = as.Date(c("2026-02-01", "2026-02-03", "2026-02-04")),
  account = c(" cash", "bank", "cash"), amount = c(12, -4.256, 0.5),
  currency = c("eur ", "usd", "EUR"), memo = c("a\nb", "", "rent"), stringsAsFactors = FALSE)
fields <- c("entry_id", "posted", "account", "amount", "currency", "memo")

out <- prepare_ledger_output(ledger)
stopifnot(identical(names(out), fields))
stopifnot(identical(out$entry_id, c("1", "2", "3")), identical(out$posted[1], "2026-02-01"))
stopifnot(identical(out$account, c("cash", "bank", "cash")))
stopifnot(identical(out$currency, c("EUR", "USD", "EUR")))
stopifnot(identical(out$amount, c("12.00", "-4.26", "0.50")), identical(out$memo[1], "a b"))
stopifnot(identical(prepare_ledger_output(ledger, digits = 0)$amount, c("12", "-4", "0")))
msg <- tryCatch(prepare_ledger_output(ledger[1:2]), error = conditionMessage)
stopifnot(identical(msg, "Ledger is missing column(s): account, amount, currency, memo"))

dir <- tempfile("ledgerw")
dir.create(dir)
p <- write_ledger_csv(ledger, file.path(dir, "l.csv"))
back <- utils::read.csv(p, colClasses = "character")
stopifnot(identical(back$amount, out$amount), !file.exists(paste0(p, ".part")))
msg <- tryCatch(write_ledger_csv(ledger, p), error = conditionMessage)
stopifnot(grepl("already exists", msg))
write_ledger_csv(ledger[1, ], p, overwrite = TRUE)
stopifnot(nrow(utils::read.csv(p)) == 1)
msg <- tryCatch(write_ledger_csv(ledger[0, ], p, overwrite = TRUE), warning = conditionMessage)
stopifnot(identical(msg, sprintf("Writing an empty ledger to `%s`.", p)))

p <- write_ledger_tsv(ledger, file.path(dir, "l.tsv"))
stopifnot(identical(readLines(p)[1], paste(fields, collapse = "\t")))
p <- write_ledger_rds(ledger, file.path(dir, "l.rds"))
stopifnot(identical(readRDS(p), out))
msg <- tryCatch(write_ledger_rds(ledger, p), error = conditionMessage)
stopifnot(grepl("already exists", msg))
p <- write_ledger_dcf(ledger, file.path(dir, "l.dcf"))
stopifnot(identical(unname(read.dcf(p)[, "amount"]), out$amount))

for (fmt in c("csv", "tsv", "rds", "dcf")) {
  target <- file.path(dir, "sub", paste0("x.", fmt))
  res <- write_ledger(ledger, target, format = fmt)
  stopifnot(identical(res, target), file.exists(target))
}

split <- write_ledger_split(ledger, file.path(dir, "split"), format = "csv")
stopifnot(identical(split$account, c("bank", "cash")),
  identical(split$file, c("bank.csv", "cash.csv")))
stopifnot(identical(split$rows, c(1L, 2L)), identical(split$total, c(-4.256, 12.5)))
stopifnot(nrow(utils::read.csv(file.path(dir, "split", "cash.csv"))) == 2)
odd <- ledger
odd$account <- c("a/b", "a/b", "c d")
split <- write_ledger_split(odd, file.path(dir, "odd"), format = "rds")
stopifnot(identical(split$file, c("a_b.rds", "c_d.rds")))
odd$account <- c("a/b", "a b", "c")
msg <- tryCatch(write_ledger_split(odd, file.path(dir, "dup")), error = conditionMessage)
stopifnot(identical(msg, "Accounts map to the same file name: a_b.csv"))
unlink(dir, recursive = TRUE)

i <- ledger_issue(c(2, 5), "debit", "missing or negative")
stopifnot(nrow(i) == 2L, identical(i$row, c(2L, 5L)), all(i$field == "debit"))
stopifnot(nrow(ledger_issue(integer(), "date", "x")) == 0L)

good <- data.frame(
  date = as.Date(c("2026-03-01", "2026-03-02")),
  account = c("1000", "4000"),
  debit = c(100, 0),
  credit = c(0, 100),
  currency = c("EUR", "EUR"),
  reference = c("J1", "J2"),
  stringsAsFactors = FALSE
)
stopifnot(nrow(validate_ledger(good)) == 0L)
codes <- good[c(1, 1, 1), ]
codes$account <- c("1200-03", "12A0", "12000")
codes$reference <- c("A", "B", "C")
stopifnot(identical(validate_ledger(codes)$row, c(2L, 3L)))
bad <- good
bad$date[1] <- NA
bad$account[2] <- "40x0"
bad$debit[2] <- -1
bad$credit[1] <- NA
bad$currency[1] <- "eu"
bad$reference[2] <- "J1"
v <- validate_ledger(bad)
stopifnot(nrow(v) == 6L)
stopifnot(identical(v$field, c("date", "credit", "currency", "account", "debit",
  "reference")))
v2 <- validate_ledger(good, period = as.Date(c("2026-03-02", "2026-03-31")))
stopifnot(nrow(v2) == 1L, v2$problem == "outside period")
stopifnot(inherits(try(validate_ledger(good[, 1:3]), silent = TRUE), "try-error"))

b <- validate_balance(good)
stopifnot(b$ok, b$debits == 100, b$difference == 0, length(b$one_sided) == 0L)
g2 <- good
g2$debit[2] <- 5
b2 <- validate_balance(g2)
stopifnot(!b2$ok, b2$difference == 5, identical(b2$one_sided, 2L))
g3 <- good
g3$debit[1] <- NA
g3$credit[2] <- -100
b3 <- validate_balance(g3)
stopifnot(identical(b3$missing$debit, 1L), length(b3$missing$credit) == 0L)
stopifnot(identical(b3$negative$credit, 2L), b3$debits == 0, b3$credits == -100, !b3$ok)

fs <- ledger_field_summary(bad)
stopifnot(identical(fs$field, c("date", "account", "debit", "credit", "currency",
  "reference")))
stopifnot(identical(fs$missing, c(1L, 0L, 0L, 1L, 0L, 0L)))
stopifnot(identical(fs$distinct, c(1L, 2L, 2L, 1L, 2L, 1L)), fs$example[1] == "2026-03-02")

messy <- data.frame(
  date = as.Date(c("2026-03-01", "2026-03-02", "2026-03-03")),
  account = c(" 1000", "4000", "4000 "),
  debit = c(10.004, NA, 0),
  credit = c(0, 10, NA),
  currency = c("eur", "EUR", " usd"),
  reference = c("J1", "", NA),
  stringsAsFactors = FALSE
)
r <- repair_ledger(messy)
stopifnot(r$changed$account == 2L, r$changed$debit == 2L, r$changed$credit == 1L)
stopifnot(r$changed$currency == 2L, r$changed$reference == 2L)
stopifnot(identical(r$ledger$reference, c("J1", "AUTO-2", "AUTO-3")))
stopifnot(identical(r$ledger$currency, c("EUR", "EUR", "USD")), r$ledger$debit[1] == 10)

stopifnot(identical(assert_ledger(good), good))
msg <- tryCatch(assert_ledger(bad, max_shown = 2L), error = conditionMessage)
stopifnot(grepl("row 1, `date`", msg), grepl("and 5 more", msg))
msg2 <- tryCatch(assert_ledger(g2), error = conditionMessage)
stopifnot(grepl("differ by 5.00", msg2))

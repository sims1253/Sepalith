# skipped: needs vctrs (format.foliokit_ledger, print.foliokit_ledger,
#   print.summary_foliokit_ledger all format through money())
stopifnot("fee" %in% ledger_types(), length(ledger_types()) == 6L)

led <- ledger(
  date = c("2026-01-05", "2026-01-02", "2026-02-01"),
  type = c("BUY", "deposit", "fee"),
  symbol = c("ACME", NA, NA),
  quantity = c(10, NA, NA),
  price = c(100, NA, NA),
  amount = c(NA, 5000, 7),
  currency = "eur"
)
stopifnot(inherits(led, "foliokit_ledger"), identical(led$currency, "EUR"))
stopifnot(identical(led$entries$type, c("deposit", "buy", "fee")))
stopifnot(all.equal(led$entries$amount, c(5000, -1000, -7)))
stopifnot(identical(ledger()$currency, "USD"), nrow(ledger()$entries) == 0L)

# validate_ledger(): bad types, non-positive and missing trade quantities
bad <- function(...) inherits(try(ledger(...), silent = TRUE), "try-error")
stopifnot(bad(date = "2026-01-01", type = "gift", amount = 1))
stopifnot(bad(date = "2026-01-01", type = "buy", symbol = "A", quantity = 0, price = 1))
stopifnot(bad(date = "2026-01-01", type = "buy", symbol = "A", price = 1))
stopifnot(bad(date = NA, type = "deposit", amount = 1))

led2 <- ledger_add(led, date = "2026-01-03", type = "withdrawal", amount = 50)
stopifnot(nrow(led2$entries) == 4L, !is.unsorted(led2$entries$date))
stopifnot(led2$entries$amount[[2]] == -50, identical(led2$currency, "EUR"))

s <- summary(led2)
stopifnot(inherits(s, "summary_foliokit_ledger"), s$n == 4L)
stopifnot(s$totals[["buy"]] == -1000, s$totals[["sell"]] == 0)
stopifnot(all.equal(s$cash, 5000 - 1000 - 7 - 50))

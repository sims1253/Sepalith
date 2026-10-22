# skipped: needs vctrs (format.foliokit_positions, print.foliokit_positions)
entries <- data.frame(
  date = as.Date(c("2026-01-02", "2026-01-05", "2026-02-10", "2026-03-15", "2026-04-01")),
  type = c("deposit", "buy", "buy", "sell", "dividend"),
  symbol = c(NA, "ACME", "GLOBX", "ACME", "ACME"),
  quantity = c(NA, 10, 5, 4, NA),
  price = c(NA, 100, 40, 120, NA),
  amount = c(5000, -1000, -200, 480, 12.5),
  stringsAsFactors = FALSE
)
led <- structure(list(entries = entries, currency = "EUR"), class = "foliokit_ledger")
ledger_types <- function() c("deposit", "withdrawal", "buy", "sell", "dividend", "fee")

pos <- ledger_positions(led, as_of = "2026-12-31")
stopifnot(inherits(pos, "foliokit_positions"), identical(pos$symbol, c("ACME", "GLOBX")))
stopifnot(all.equal(pos$quantity, c(6, 5)), all.equal(pos$cost, c(600, 200)))
stopifnot(nrow(ledger_positions(led, as_of = "2026-01-01")) == 0L)

pos_bad <- new_positions(data.frame(symbol = "X", quantity = -1, cost = 0))
stopifnot(inherits(try(validate_positions(pos_bad), silent = TRUE), "try-error"))

valued <- value_positions(pos, c(ACME = 110, GLOBX = 50))
stopifnot(all.equal(valued$value, c(660, 250)), all.equal(sum(valued$weight), 1))
stopifnot(inherits(try(value_positions(pos, c(ACME = 1)), silent = TRUE), "try-error"))

df <- as.data.frame(valued)
stopifnot(identical(class(df), "data.frame"), identical(unique(df$currency), "EUR"))

cash <- ledger_cash_by_month(led)
stopifnot(identical(cash$month, c("2026-01", "2026-04")), all.equal(cash$net, c(5000, 12.5)))
stopifnot(nrow(ledger_cash_by_month(led, "withdrawal")) == 0L)

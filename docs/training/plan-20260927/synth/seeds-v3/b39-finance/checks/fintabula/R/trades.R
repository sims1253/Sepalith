tr <- new.trades(c("2026-05-03", "2026-05-01", "2026-05-02"), c("sell", "buy", "buy"),
  c(15, 10, 10), c(12, 10, 11), fee = 1)
stopifnot(identical(tr$side, c("buy", "buy", "sell")), all(tr$fee == 1))
stopifnot(inherits(try(new.trades("2026-05-01", "buy", 0, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.trades("2026-05-01", "buy", 1, NA), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.trades("2026-05-01", "buy", 1, 1, -1), silent = TRUE), "try-error"))
stopifnot(inherits(try(new.trades("2026-05-01", "hold", 1, 1), silent = TRUE), "try-error"))
msg <- tryCatch(new.trades(NA, "buy", 1, 1), error = conditionMessage)
stopifnot(msg == "`date` has invalid values.", identical(rownames(tr), c("1", "2", "3")))

stopifnot(identical(trade.cash.flows(tr), c(-101, -111, 179)))

pos <- list(qty = 0, avg.cost = 0, realized = 0, bought = 0, sold = 0, fees = 0)
pos <- apply.trade(pos, tr[1, ])
stopifnot(pos$qty == 10, pos$avg.cost == 10.1)
pos <- apply.trade(pos, tr[2, ])
stopifnot(pos$qty == 20, pos$avg.cost == 10.6)
pos <- apply.trade(pos, tr[3, ])
stopifnot(pos$qty == 5, isTRUE(all.equal(pos$realized, 179 - 15 * 10.6)))
pos <- apply.trade(pos, new.trades("2026-05-04", "sell", 5, 10))
stopifnot(pos$qty == 0, pos$avg.cost == 0, pos$bought == 20, pos$sold == 20, pos$fees == 3)
stopifnot(inherits(try(apply.trade(pos, tr[3, ]), silent = TRUE), "try-error"))

ff <- fifo.realized(tr)
stopifnot(isTRUE(all.equal(ff$realized, 10 * (12 - 1 / 15 - 10.1) + 5 * (12 - 1 / 15 - 11.1))))
stopifnot(nrow(ff$lots) == 1L, ff$lots$qty == 5, ff$matched == 2L)
stopifnot(inherits(try(fifo.realized(tr[3, ]), silent = TRUE), "try-error"))

book <- list(
  bids = data.frame(price = c(99, 98, 97), qty = c(5, 5, 5)),
  asks = data.frame(price = c(102, 101, 103), qty = c(5, 5, 5))
)
b <- fill.order("buy", 8, book)
stopifnot(b$filled == 8, b$unfilled == 0, b$levels == 2L, b$avg.price == (5 * 101 + 3 * 102) / 8)
s <- fill.order("sell", 20, book)
stopifnot(s$filled == 15, s$unfilled == 5, s$levels == 3L, s$avg.price == 98)
stopifnot(is.na(fill.order("sell", 1, list(bids = book$bids[0, ]))$avg.price))

tt <- new.trades(c("2026-05-01", "2026-05-02"), "buy", c(100, 1000), c(5, 20))
stopifnot(identical(apply.fee.model(tt, "flat")$fee, c(5, 5)))
stopifnot(identical(apply.fee.model(tt, "per.share")$fee, c(1, 10)))
stopifnot(identical(apply.fee.model(tt, "percent", minimum = 0)$fee, c(0.5, 20)))
stopifnot(identical(apply.fee.model(tt, "tiered")$fee, c(1, 15)))

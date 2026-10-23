stopifnot(identical(account_class(c("1000", "2100", "3100", "4000", "5000", "6100", "9000")),
  c("asset", "liability", "equity", "revenue", "expense", "expense", NA)))

stopifnot(identical(normal_side("asset"), list(side = "debit", sign = 1,
  statement = "balance", temporary = FALSE)))
stopifnot(normal_side("liability")$side == "credit", normal_side("equity")$sign == -1)
stopifnot(normal_side("revenue")$side == "credit", normal_side("expense")$sign == 1)
stopifnot(normal_side("revenue")$temporary, normal_side("expense")$statement == "income")
stopifnot(!normal_side("liability")$temporary, normal_side("equity")$statement == "balance")

ch <- new_chart(c("4000", "1000", "2000", "3100", "6000"),
  c("Sales", "Cash", "Payables", "Retained earnings", "Rent"))
stopifnot(identical(ch$code, c("1000", "2000", "3100", "4000", "6000")))
stopifnot(identical(ch$side, c("debit", "credit", "credit", "credit", "debit")))
stopifnot(inherits(try(new_chart(c("1000", "1000"), c("a", "b")), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(new_chart("1000", ""), silent = TRUE), "try-error"))
stopifnot(inherits(try(new_chart("9000", "x"), silent = TRUE), "try-error"))

cs <- chart_summary(ch)
stopifnot(identical(cs$class, c("asset", "liability", "equity", "revenue", "expense")))
stopifnot(identical(cs$n, rep(1L, 5L)), cs$first[1] == "1000", cs$last[5] == "6000")
cs2 <- chart_summary(new_chart(c("1000", "1100", "6000"), c("Cash", "Bank", "Rent")))
stopifnot(identical(cs2$n, c(2L, 0L, 0L, 0L, 1L)), cs2$last[1] == "1100", is.na(cs2$first[2]))

ent <- data.frame(
  account = c("1000", "4000", "6000", "1000", "2000", "6000"),
  debit = c(500, 0, 120, 0, 0, 30),
  credit = c(0, 500, 0, 120, 30, 0),
  stringsAsFactors = FALSE
)
tb <- trial_balance(ent, ch)
stopifnot(isTRUE(attr(tb, "balanced")))
stopifnot(identical(tb$balance, c(380, 30, 0, 500, 150)))
stopifnot(!isTRUE(attr(trial_balance(ent[-1, ], ch), "balanced")))

ce <- closing_entries(tb, "2026-12-31")
stopifnot(nrow(ce) == 3L, identical(ce$account, c("4000", "6000", "3100")))
stopifnot(ce$debit[1] == 500, ce$credit[2] == 150, ce$credit[3] == 350, ce$debit[3] == 0)
stopifnot(sum(ce$debit) == sum(ce$credit))
tb2 <- tb
tb2$balance[tb2$code == "6000"] <- 800
ce2 <- closing_entries(tb2, "2026-12-31", retained = "3200")
stopifnot(ce2$account[3] == "3200", ce2$debit[3] == 300)

b <- new_posting_batch("2026-01-05", c("cash", "revenue", "payables", "inventory"),
  c(100, 100, 40, 40), c("debit", "credit", "credit", "debit"))
stopifnot(identical(b$side, c("debit", "credit", "credit", "debit")))
stopifnot(nrow(b) == 4L, all(b$date == as.Date("2026-01-05")), all(b$memo == ""))
stopifnot(inherits(try(new_posting_batch("2026-01-05", "cash", -1, "debit"),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(new_posting_batch(c("2026-01-05", "2026-01-06"), "cash",
  c(1, 2, 3), "debit"), silent = TRUE), "try-error"))
stopifnot(inherits(try(new_posting_batch("2026-01-05", "cash", 1, "both"),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(new_posting_batch("2026-01-05", NA, 1, "debit"),
  silent = TRUE), "try-error"))
stopifnot(inherits(try(new_posting_batch("2026-01-05", "cash", c(1, 2), "debit",
  memo = c("a", "b", "c")), silent = TRUE), "try-error"))

bal <- list(cash = 10, receivables = 5, inventory = 0, payables = 0, revenue = 0)
b2 <- rbind(b, new_posting_batch("2026-01-06", c("receivables", "cash"), c(5, 5),
  c("credit", "debit")))
res <- apply_postings(bal, b2)
stopifnot(res$balances$cash == 115, res$balances$receivables == 0)
stopifnot(res$balances$inventory == 40, res$balances$payables == 40,
  res$balances$revenue == 100)
stopifnot(res$posted$cash == 2L, res$posted$revenue == 1L)
w <- tryCatch(apply_postings(bal, new_posting_batch("2026-01-01", "misc", 1, "debit")),
  warning = function(w) conditionMessage(w))
stopifnot(grepl("misc", w))

s <- posting_summary(b2)
stopifnot(s$n == 6L, s$debits$total == 145, s$credits$total == 145, s$net == 0)
stopifnot(isTRUE(s$balanced), s$debits$n == 3L, s$credits$largest == 100)
stopifnot(identical(s$debits$accounts, c("cash", "inventory")))
stopifnot(identical(s$credits$accounts, c("payables", "receivables", "revenue")))
stopifnot(s$first == as.Date("2026-01-05"), s$last == as.Date("2026-01-06"))
s1 <- posting_summary(b[1, ])
stopifnot(!s1$balanced, s1$credits$largest == 0, s1$credits$n == 0L)

led <- data.frame(date = as.Date(character()), account = character(), debit = numeric(),
  credit = numeric(), balance = numeric(), memo = character(), stringsAsFactors = FALSE)
stopifnot(last_balance(led, "cash") == 0)
led <- post_credit_purchase(led, 80, "2026-02-01")
stopifnot(nrow(led) == 2L, identical(led$account, c("inventory", "payables")))
stopifnot(identical(led$balance, c(80, 80)), led$credit[2] == 80)
led <- post_credit_sale(led, 120, "2026-02-02", "Invoice 7")
stopifnot(identical(led$balance[3:4], c(120, 120)), all(led$memo[3:4] == "Invoice 7"))
led <- post_cash_receipt(led, 100, "2026-02-03")
stopifnot(identical(led$account[5:6], c("cash", "receivables")))
stopifnot(identical(led$balance[5:6], c(100, 20)), last_balance(led, "receivables") == 20)
led <- post_cash_payment(led, 30, "2026-02-04")
stopifnot(identical(led$account[7:8], c("payables", "cash")))
stopifnot(identical(led$balance[7:8], c(50, 70)), led$debit[7] == 30, led$credit[8] == 30)
stopifnot(inherits(try(post_cash_payment(led, 0, "2026-02-05"), silent = TRUE), "try-error"))
stopifnot(inherits(try(post_credit_sale(led, NA_real_, "2026-02-05"), silent = TRUE),
  "try-error"))

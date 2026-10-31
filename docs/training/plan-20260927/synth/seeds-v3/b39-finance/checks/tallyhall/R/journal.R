je <- journal_entry("1000", "4000", 250, "2026-04-01", "Sale")
stopifnot(nrow(je) == 2L, identical(je$debit, c(250, 0)), identical(je$credit, c(0, 250)))
stopifnot(all(je$memo == "Sale"), identical(je$account, c("1000", "4000")))
stopifnot(inherits(try(journal_entry("1000", "1000", 1, "2026-04-01"), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(journal_entry("1000", "4000", -1, "2026-04-01"), silent = TRUE),
  "try-error"))

rv <- reverse_entry(je, "2026-04-30")
stopifnot(identical(rv$debit, c(0, 250)), identical(rv$credit, c(250, 0)))
stopifnot(all(rv$date == as.Date("2026-04-30")), rv$memo[1] == "Reversal: Sale")
stopifnot(all(reverse_entry(je)$date == as.Date("2026-04-01")))

st <- side_totals(rbind(je, journal_entry("5000", "1000", 100, "2026-04-02")))
stopifnot(identical(st$account, c("1000", "4000", "5000")))
stopifnot(identical(st$net, c(150, -250, 100)))
stopifnot(identical(st$side, c("debit", "credit", "debit")))
stopifnot(side_totals(rbind(je, reverse_entry(je)))$side[1] == "zero")

acc <- list(expense = "6000", revenue = "4000", accrued_asset = "1300",
  accrued_liability = "2100", prepaid_asset = "1400", deferred_liability = "2400")
a1 <- adjusting_entry("accrued_expense", 10, "2026-04-30", acc)
stopifnot(identical(a1$account, c("6000", "2100")), a1$memo[1] == "Adjustment: accrued expense")
a2 <- adjusting_entry("accrued_revenue", 10, "2026-04-30", acc)
stopifnot(identical(a2$account, c("1300", "4000")))
a3 <- adjusting_entry("prepaid_expense", 10, "2026-04-30", acc)
stopifnot(identical(a3$account, c("6000", "1400")))
a4 <- adjusting_entry("deferred_revenue", 10, "2026-04-30", acc)
stopifnot(identical(a4$account, c("2400", "4000")), sum(a4$credit) == 10)

sp <- split_entry("6000", c("2000", "2100", "2200"), c(1, 1, 1), 100, "2026-04-30")
stopifnot(nrow(sp) == 4L, sp$debit[1] == 100, isTRUE(all.equal(sum(sp$credit), 100)))
stopifnot(isTRUE(all.equal(sp$credit[2:4], c(33.33, 33.33, 33.34))))
stopifnot(inherits(try(split_entry("6000", "2000", c(1, 2), 1, "2026-04-30"),
  silent = TRUE), "try-error"))

fe <- format_entry(je)
stopifnot(length(fe) == 5L, startsWith(fe[1], "Date"), grepl("^2026-04-01  Dr 1000", fe[2]))
stopifnot(grepl("^2026-04-01      Cr 4000", fe[3]), grepl("250.00$", fe[3]))
stopifnot(length(unique(nchar(fe))) == 1L, grepl("^Total .*250.00 +250.00$", fe[5]))
ft <- format_entry(je, indent = 1L, title = "Sales journal")
stopifnot(ft[1] == "Sales journal", grepl("^2026-04-01   Cr", ft[4]))

# skipped: categoryBudgetUI, categoryBudgetServer (need shiny / bslib);
#   plotVariance (needs ggplot2)

tx <- data.frame(
  date = as.Date(c("2026-09-02", "2026-09-03", "2026-09-15", NA, "2026-10-01")),
  category = c("groceries", "rent", "groceries", "fun", "rent"),
  amount = c(-120, -900, -80, -45, -900)
)

# spendByCategory skips rows with a missing date instead of returning NA rows
spent <- spendByCategory(tx, "2026-09")
stopifnot(identical(spent, c(groceries = 200, rent = 900)))
stopifnot(length(spendByCategory(tx, "2026-07")) == 0L)

# budgetVariance and varianceStatus
v <- budgetVariance(spent, c(groceries = 250, rent = 900, fun = 40))
stopifnot(identical(v$spent, c(200, 900, 0)), identical(v$remaining, c(50, 0, 40)))
stopifnot(identical(v$status, c("under", "near", "under")))
stopifnot(identical(varianceStatus(c(10, 95, 101), 100), c("under", "near", "over")))

# formatMoney and formatVarianceTable
stopifnot(identical(formatMoney(c(1234.5, -0.25)), c("$1,234.50", "-$0.25")))
stopifnot(identical(formatMoney(3, symbol = "EUR "), "EUR 3.00"))
ft <- formatVarianceTable(v)
stopifnot(identical(names(ft), c("Category", "Limit", "Spent", "Remaining", "Status")))
stopifnot(identical(ft$Limit[1], "$250.00"))

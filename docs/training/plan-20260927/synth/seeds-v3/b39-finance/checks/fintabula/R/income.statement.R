tb <- data.frame(
  code = c("4000", "4100", "5000", "6100", "6200", "7100", "8100"),
  balance = c(900, 100, 400, 150, 50, 20, 76),
  stringsAsFactors = FALSE
)
stopifnot(line.amount(tb, "4") == 1000, line.amount(tb, c("61", "62")) == 200)
stopifnot(line.amount(tb, "9") == 0)

map <- list(revenue = "4", cost.of.sales = "5", operating.expenses = "6",
  interest.expense = "71", tax = "81")
is <- build.income.statement(tb, map)
stopifnot(inherits(is, "fin.statement"))
stopifnot(identical(names(is), c("revenue", "cost.of.sales", "gross.profit",
  "operating.expenses", "operating.income", "interest.expense", "pretax.income", "tax",
  "net.income")))
stopifnot(is[["gross.profit"]] == 600, is[["operating.income"]] == 400)
stopifnot(is[["pretax.income"]] == 380, is[["net.income"]] == 304)

mr <- margin.ratios(is)
stopifnot(mr[["gross.margin"]] == 0.6, mr[["operating.margin"]] == 0.4)
stopifnot(mr[["pretax.margin"]] == 0.38, mr[["tax.rate"]] == 0.2, mr[["net.margin"]] == 0.304)
stopifnot(margin.ratios(is, digits = 1L)[["net.margin"]] == 0.3)
empty <- margin.ratios(build.income.statement(tb[0, ], map))
stopifnot(all(is.na(empty)), length(empty) == 5L)

prior <- build.income.statement(transform(tb, balance = balance / 2), map)
v <- statement.variance(is, prior)
stopifnot(nrow(v) == 9L, v$change[1] == 500, all(v$pct.change == 100), all(v$flag))
v2 <- statement.variance(is, is, threshold = 0)
stopifnot(!any(v2$flag), all(v2$change == 0))
z <- prior
z[["tax"]] <- 0
v3 <- statement.variance(is, z, threshold = 1000)
stopifnot(is.na(v3$pct.change[8]), v3$flag[8], !v3$flag[1])
stopifnot(inherits(try(statement.variance(is, prior[1:3]), silent = TRUE), "try-error"))

f <- format(is)
stopifnot(length(f) == 13L, startsWith(f[1], "Revenue"), grepl("1,000.00$", f[1]))
stopifnot(f[3] == strrep("-", 40L), startsWith(f[13], "Net income"))
ft <- format(is, title = "Income statement")
stopifnot(length(ft) == 15L, ft[1] == "            Income statement", ft[2] == strrep("=", 40L))
neg <- structure(c(total.loss = -5), class = c("fin.statement", "numeric"))
stopifnot(grepl("\\(5.00\\)$", format(neg)[2]))
out <- capture.output(r <- print(is))
stopifnot(identical(out, f), identical(r, is))

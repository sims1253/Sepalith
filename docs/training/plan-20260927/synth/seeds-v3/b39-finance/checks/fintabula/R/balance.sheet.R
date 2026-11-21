# line.amount() lives in income.statement.R
line.amount <- function(tb, prefixes) {
  hit <- Reduce(`|`, lapply(prefixes, function(p) startsWith(tb$code, p)), FALSE)
  sum(tb$balance[hit])
}

tb <- data.frame(
  code = c("1000", "1100", "1200", "1500", "2000", "2100", "2500", "3000"),
  balance = c(50, 30, 20, 400, 40, 10, 150, 300),
  stringsAsFactors = FALSE
)
map <- list(cash = "10", receivables = "11", inventory = "12", fixed.assets = "15",
  payables = "20", short.term.debt = "21", long.term.debt = "25", equity = "3")
bs <- build.balance.sheet(tb, map)
stopifnot(inherits(bs, "fin.statement"), length(bs) == 13L)
stopifnot(bs[["total.current.assets"]] == 100, bs[["total.assets"]] == 500)
stopifnot(bs[["total.current.liabilities"]] == 50, bs[["total.liabilities"]] == 200)
stopifnot(bs[["total.liabilities.equity"]] == 500)

stopifnot(length(check.balance.sheet(bs)) == 0L)
bad <- bs
bad[["total.current.assets"]] <- 90
bad[["total.current.liabilities"]] <- 60
p <- check.balance.sheet(bad)
stopifnot(length(p) == 4L, p[1] == "`total.current.assets` should be 100.00")
stopifnot(p[2] == "`total.assets` should be 490.00",
  p[3] == "`total.current.liabilities` should be 50.00")
bad2 <- bs
bad2[["equity"]] <- 290
bad2[["total.liabilities.equity"]] <- 490
p2 <- check.balance.sheet(bad2)
stopifnot(length(p2) == 1L, grepl("differ by 10.00", p2))
bad3 <- bs
bad3[["total.assets"]] <- 1
bad3[["total.liabilities"]] <- 2
stopifnot(length(check.balance.sheet(bad3)) == 4L)

lr <- liquidity.ratios(bs)
stopifnot(liquidity.ratios(bs, 1L)[["debt.to.equity"]] == 0.5)
stopifnot(lr[["current.ratio"]] == 2, lr[["quick.ratio"]] == 1.6, lr[["cash.ratio"]] == 1)
stopifnot(lr[["debt.to.equity"]] == 0.5333, lr[["equity.ratio"]] == 0.6)
stopifnot(is.na(liquidity.ratios(build.balance.sheet(tb[1, ], map))[["current.ratio"]]))

prior <- build.balance.sheet(transform(tb, balance = balance / 2), map)
we <- working.capital.effect(bs, prior)
stopifnot(identical(we$item, c("receivables", "inventory", "payables")))
stopifnot(identical(we$type, c("asset", "asset", "liability")))
stopifnot(identical(we$effect, c(-15, -10, 20)), attr(we, "total") == -5)
stopifnot(identical(we$change, c(15, 10, 20)), we$prior[1] == 15)
stopifnot(attr(working.capital.effect(bs, prior, "cash", character()), "total") == -25)

cl <- classify.accounts(tb, map)
stopifnot(identical(cl$line[1:3], c("cash", "receivables", "inventory")), cl$line[8] == "equity")
stopifnot(is.na(classify.accounts(tb, map[1:2])$line[4]))
w <- tryCatch(classify.accounts(tb, c(map, list(all.current = "1"))), warning = conditionMessage)
stopifnot(grepl("1000", w))

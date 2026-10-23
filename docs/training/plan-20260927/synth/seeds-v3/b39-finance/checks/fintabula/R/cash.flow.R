is <- c(net.income = 120)
prior <- c(cash = 100, receivables = 50, inventory = 40, payables = 30,
  short.term.debt = 10, long.term.debt = 100)
current <- c(cash = 175, receivables = 70, inventory = 30, payables = 45,
  short.term.debt = 20, long.term.debt = 80)
cf <- build.cash.flow(is, current, prior, depreciation = 25, capex = 60, dividends = 15)
stopifnot(inherits(cf, "fin.statement"), length(cf) == 12L, names(cf)[12] == "net.change")
stopifnot(cf[["change.receivables"]] == -20, cf[["change.inventory"]] == 10)
stopifnot(cf[["change.payables"]] == 15, cf[["operating"]] == 150)
stopifnot(cf[["investing"]] == -60, cf[["debt.change"]] == -10, cf[["financing"]] == -25)
stopifnot(cf[["net.change"]] == 65)

rc <- reconcile.cash(cf, current, prior)
stopifnot(!rc$ok, rc$difference == 10, rc$opening == 100, rc$closing == 175)
current2 <- current
current2[["cash"]] <- 165
stopifnot(reconcile.cash(cf, current2, prior)$ok)

stopifnot(free.cash.flow(cf) == 90)
stopifnot(free.cash.flow(cf, "firm", interest = 10, tax.rate = 0.25) == 97.5)
stopifnot(free.cash.flow(cf, "equity") == 80)
stopifnot(free.cash.flow(cf, "owner") == 85)
stopifnot(free.cash.flow(cf, "levered", interest = 10, tax.rate = 0.25) == 72.5)

tx <- data.frame(
  category = c("customers", "customers", "suppliers", "employees", "interest", "taxes"),
  amount = c(500, 200, -300, -150, -20, -40),
  stringsAsFactors = FALSE
)
d <- direct.cash.flow(tx)
stopifnot(identical(names(d), c("customers", "suppliers", "employees", "interest",
  "taxes", "operating")))
stopifnot(d[["customers"]] == 700, d[["suppliers"]] == -300, d[["operating"]] == 190)
stopifnot(d[["employees"]] == -150, d[["interest"]] == -20, d[["taxes"]] == -40)
w <- tryCatch(direct.cash.flow(rbind(tx, data.frame(category = "loan", amount = 5))),
  warning = conditionMessage)
stopifnot(grepl("loan", w))

pc <- project.cash(100, c(10, 0, 50), c(30, 90, 0), floor = 0, ceiling = 50)
stopifnot(identical(pc$balance, c(80, -10, 40)), identical(pc$below, 2L))
stopifnot(identical(pc$above, 1L), pc$highest$gap == -30)
stopifnot(pc$lowest$period == 2L, pc$lowest$value == -10, pc$lowest$gap == -10)
stopifnot(pc$highest$period == 1L, pc$highest$value == 80)
stopifnot(length(project.cash(0, 1, 0)$above) == 0L, project.cash(0, 1, 0)$highest$gap == Inf)
stopifnot(inherits(try(project.cash(0, 1, c(1, 2)), silent = TRUE), "try-error"))
stopifnot(inherits(try(project.cash(0, 1, 1, floor = 5, ceiling = 1), silent = TRUE),
  "try-error"))

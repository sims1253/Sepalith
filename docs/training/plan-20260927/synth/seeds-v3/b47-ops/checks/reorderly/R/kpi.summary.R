sim <- data.frame(demand = c(10, 10, 10, 10), on.hand = c(5, 0, 0, 5),
  backorder = c(0, 3, 1, 0), order = c(0, 20, 0, 0))
k <- kpi.summary(sim)
stopifnot(isTRUE(all.equal(k[["fill.rate"]], 0.925)), k[["cycle.service"]] == 0.75)
stopifnot(k[["avg.inventory"]] == 2.5, k[["turns"]] == 16)
stopifnot(k[["stockout.periods"]] == 2, k[["avg.backorder"]] == 1)
stopifnot(identical(names(kpi.summary(sim, c("turns", "fill.rate"))), c("turns", "fill.rate")))
empty <- data.frame(demand = c(0, 0), on.hand = c(0, 0), backorder = c(0, 0), order = 0)
ke <- kpi.summary(empty)
stopifnot(ke[["fill.rate"]] == 1, is.infinite(ke[["turns"]]))

tc <- kpi.targets.check(k, c(fill.rate = 0.95, avg.inventory = 3, turns = 10))
stopifnot(identical(tc$kpi, c("fill.rate", "avg.inventory", "turns")))
stopifnot(identical(tc$met, c(FALSE, TRUE, TRUE)), isTRUE(all.equal(tc$gap, c(-0.025, 0.5, 6))))
tc2 <- kpi.targets.check(k, c(stockout.periods = 1, avg.backorder = 1))
stopifnot(identical(tc2$met, c(FALSE, TRUE)), identical(tc2$gap, c(-1, 0)))
stopifnot(inherits(try(kpi.targets.check(k, c(other = 1)), silent = TRUE), "try-error"))

rt <- render.kpi.table(k)
stopifnot(identical(rt$text, c("92.5%", "75.0%", "2.5 units", "16.0x", "2", "1.0 units")))
stopifnot(rt$label[4] == "Inventory turns", nrow(rt) == 6)
stopifnot(identical(render.kpi.table(k[c("turns", "fill.rate")], 0)$text, c("16x", "92%")))

w <- character(0)
pc <- withCallingHandlers(kpi.period.compare(c(a = 2, b = 0, c = 5), c(a = 3, b = 1, d = 1)),
  warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
stopifnot(identical(pc$kpi, c("a", "b")), identical(pc$change, c(1, 1)))
stopifnot(pc$pct.change[1] == 50, is.na(pc$pct.change[2]))
stopifnot(length(w) == 2, grepl("`before`: c", w[1]), grepl("`after`: d", w[2]))

tab <- data.frame(item = c("x", "y", "z"), fill.rate = c(0.9, 1, 0.8), turns = c(10, 5, 10),
  avg.inventory = c(5, 10, 5), stockout.periods = c(1, 0, 2))
sc <- kpi.scorecard(tab)
stopifnot(identical(sc$item, c("x", "y", "z")), isTRUE(all.equal(sc$score, c(0.7, 0.6, 0.4))))
sc2 <- kpi.scorecard(tab, weights = c(fill.rate = 1, turns = 0, avg.inventory = 0,
  stockout.periods = 0))
stopifnot(identical(sc2$item, c("y", "x", "z")), sc2$score[1] == 1)

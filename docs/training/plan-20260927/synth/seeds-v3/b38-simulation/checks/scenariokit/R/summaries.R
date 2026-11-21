err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
r1 <- data.frame(time = 0:4, S = c(95, 90, 80, 75, 75), I = c(5, 8, 12, 6, 0),
  R = c(0, 2, 8, 19, 25), incidence = c(0, 5, 10, 5, 0))
r2 <- data.frame(time = 0:4, S = c(95, 93, 90, 88, 87), I = c(5, 6, 5, 3, 2),
  R = c(0, 1, 5, 9, 11), incidence = c(0, 2, 3, 2, 1))
runs <- list(a = r1, b = r2)

stopifnot(identical(metric.peak.size(runs), c(a = 12, b = 6)))
stopifnot(identical(metric.peak.time(runs), c(a = 2, b = 1)))
stopifnot(identical(metric.attack.rate(runs), c(a = 0.2, b = 0.08)))
stopifnot(identical(metric.duration(runs), c(a = 3, b = 4)))
stopifnot(identical(metric.peak.size(runs, "R"), c(a = 25, b = 11)))
stopifnot(err(metric.peak.time(list())) == "`runs` must be a non-empty list of trajectories.")
stopifnot(err(metric.duration(runs, "E")) == "Column `E` not found in a run.")
stopifnot(err(metric.attack.rate(runs, "cases")) == "Column `cases` not found in a run.")

sm <- summarise.metric(c(1, 2, 3, 4, NA))
stopifnot(identical(names(sm), c("mean", "sd", "median", "lower", "upper")))
stopifnot(sm[["mean"]] == 2.5, sm[["median"]] == 2.5, isTRUE(all.equal(sm[["lower"]], 1.15)))
stopifnot(isTRUE(all.equal(sm[["upper"]], 3.85)), isTRUE(all.equal(sm[["sd"]], sd(1:4))))
stopifnot(is.na(summarise.metric(5, "sd")[["sd"]]), identical(names(summarise.metric(1:3,
  "upper")), "upper"))

tab <- summarise.replicates(runs)
stopifnot(identical(tab$metric, c("peak.size", "peak.time", "attack.rate", "duration")))
stopifnot(tab$mean[1] == 9, tab$mean[2] == 1.5, isTRUE(all.equal(tab$mean[3], 0.14)),
  tab$mean[4] == 3.5)
stopifnot(nrow(summarise.replicates(runs, "duration")) == 1L)

cmp <- compare.scenarios(list(x = runs, y = list(r1)))
stopifnot(nrow(cmp) == 8L, identical(unique(cmp$scenario), c("x", "y")))
stopifnot(is.na(cmp$sd[cmp$scenario == "y"][1]), names(cmp)[1:2] == c("scenario", "metric"))
stopifnot(nchar(err(compare.scenarios(list(runs)))) > 0)
r3 <- r1; r3$I[2] <- NA
stopifnot(is.na(metric.peak.size(list(r3))), is.na(metric.duration(list(r3))))
stopifnot(is.na(metric.peak.time(list(r3))), !is.na(metric.attack.rate(list(r3))))

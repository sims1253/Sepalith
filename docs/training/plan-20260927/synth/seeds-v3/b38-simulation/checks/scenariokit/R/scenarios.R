set.seed(2024)
base <- list(beta = 0.4, gamma = 0.1, seed.cases = 5L)
ep <- toy.epidemic(base, days = 60, N = 500)
stopifnot(nrow(ep) == 61L, identical(names(ep), c("time", "S", "I", "R", "incidence")))
stopifnot(all(ep$S + ep$I + ep$R == 500), ep$I[1] == 5, all(diff(ep$S) == -ep$incidence[-1]))
ep2 <- toy.epidemic(list(beta = 0.4, gamma = 0.1, immune = 0.2), days = 5, N = 100)
stopifnot(ep2$R[1] == 20, ep2$I[1] == 5)

sc <- define.scenarios(base)
stopifnot(identical(names(sc), c("baseline", "lockdown", "vaccination", "school.closure",
  "combined")))
stopifnot(sc$baseline$beta == 0.4, sc$lockdown$beta == 0.2,
  isTRUE(all.equal(sc$school.closure$beta, 0.32)))
stopifnot(isTRUE(all.equal(sc$vaccination$immune, 0.4)), sc$vaccination$beta == 0.4)
stopifnot(sc$combined$beta == 0.2, isTRUE(all.equal(sc$combined$immune, 0.4)))
stopifnot(identical(vapply(sc, `[[`, "", "label"), setNames(names(sc), names(sc))))
stopifnot(identical(names(define.scenarios(base, c("lockdown", "baseline"))),
  c("baseline", "lockdown")))

gr <- scenario.grid(beta = c(0.3, 0.5), gamma = 0.1, immune = c(0, 0.2))
stopifnot(length(gr) == 4L, names(gr)[1] == "beta=0.3/gamma=0.1/immune=0")
stopifnot(gr[[4]]$beta == 0.5, gr[[4]]$immune == 0.2)

res <- run.policy.experiment(base, n.reps = 3L, days = 30L, seed = 7L)
stopifnot(identical(names(res), names(sc)), all(lengths(res) == 3L))
set.seed(7L)
stopifnot(identical(res$baseline[[1]], toy.epidemic(sc$baseline, 30L)))
set.seed(7L)
stopifnot(identical(res$combined[[1]], toy.epidemic(sc$combined, 30L)))

gres <- run.grid.experiment(gr, n.reps = 2L, days = 10L)
stopifnot(identical(names(gres), names(gr)), all(lengths(gres) == 2L))
stopifnot(inherits(try(run.grid.experiment(list()), silent = TRUE), "try-error"))

ct <- scenario.contrast(res, function(run) max(run$I))
stopifnot(identical(names(ct), c("scenario", "mean", "se", "difference", "ratio", "paired.se")))
stopifnot(ct$difference[1] == 0, ct$ratio[1] == 1, nrow(ct) == 5L, ct$paired.se[1] == 0)
pk <- vapply(res$lockdown, function(run) max(run$I), numeric(1))
stopifnot(isTRUE(all.equal(ct$mean[2], mean(pk))), isTRUE(all.equal(ct$se[2], sd(pk) / sqrt(3))))
stopifnot(inherits(try(scenario.contrast(res, max, "none"), silent = TRUE), "try-error"))

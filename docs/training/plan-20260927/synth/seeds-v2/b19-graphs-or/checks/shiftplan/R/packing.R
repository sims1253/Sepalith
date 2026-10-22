# skipped: needs lpSolve (knapsack.lp)
v <- c(60, 100, 120, 30)
w <- c(10L, 20L, 30L, 5L)

# knapsack.dp
dp <- knapsack.dp(v, w, 50L)
stopifnot(dp$value == 220, identical(which(dp$chosen), c(2L, 3L)))
stopifnot(knapsack.dp(v, w, 0L)$value == 0)
stopifnot(identical(knapsack.dp(numeric(0), integer(0), 5L)$chosen, logical(0)))

# knapsack.greedy
gr <- knapsack.greedy(v, w, 50L)
stopifnot(gr$weight <= 50, gr$value >= dp$value / 2)
stopifnot(identical(which(knapsack.greedy(c(1, 1, 10), c(1, 1, 10), 10)$chosen), 3L))

# binpack.first.fit / binpack.best.fit
size <- c(4, 8, 1, 4, 2, 1, 7, 3)
ff <- binpack.first.fit(size, 10)
bf <- binpack.best.fit(size, 10)
stopifnot(all(tapply(size, ff, sum) <= 10), all(tapply(size, bf, sum) <= 10))
stopifnot(max(ff) == 3L, max(bf) == 3L)
stopifnot(inherits(try(binpack.first.fit(c(3, 12), 10), silent = TRUE), "try-error"))

# binpack.lower.bound / binpack.summary
stopifnot(binpack.lower.bound(size, 10) == 3L, binpack.lower.bound(c(6, 6, 5, 3), 10) == 2L)
sm <- binpack.summary(size, ff, 10)
stopifnot(sum(sm$load) == sum(size), attr(sm, "gap") == 0L)

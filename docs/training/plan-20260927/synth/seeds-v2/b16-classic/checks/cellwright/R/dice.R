fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
set.seed(42)

# check.count / check.probability
stopifnot(check.count(3, "n") == 3, fails(check.count(0, "n")), fails(check.count(2.5, "n")))
stopifnot(check.probability(0.2, "p") == 0.2, fails(check.probability(1.5, "p")))
stopifnot(fails(check.probability(NA_real_, "p")))

# roll.dice
r = roll.dice(3, 6, 10)
stopifnot(identical(dim(r), c(10L, 3L)), all(r >= 1 & r <= 6), is.integer(r))

# throw.totals / throw.doubles
m = rbind(c(1L, 1L), c(2L, 5L))
stopifnot(identical(throw.totals(m), c(2L, 7L)), identical(throw.totals(c(3, 4)), 7L))
stopifnot(identical(throw.doubles(m), c(TRUE, FALSE)))

# tally.faces
tf = tally.faces(m, 6)
stopifnot(identical(unname(tf), c(2L, 1L, 0L, 0L, 1L, 0L)))
stopifnot(identical(names(tf), as.character(1:6)))
stopifnot(fails(tally.faces(m, 4)))

# craps.round
cr = craps.round()
stopifnot(cr$outcome %in% c("win", "lose"), cr$throws >= 1L)

# pig.turn
pt = pig.turn(hold.at = 10)
stopifnot(pt$score == 0 || pt$score >= 10, length(pt$rolls) >= 1)
stopifnot(pt$score == 0 || !1L %in% pt$rolls)

# total.chance: P(total of 7 with two dice) = 1/6
tc = total.chance(7, n.throws = 20000)
stopifnot(identical(names(tc), c("estimate", "lower", "upper")))
stopifnot(abs(tc[["estimate"]] - 1 / 6) < 0.02)

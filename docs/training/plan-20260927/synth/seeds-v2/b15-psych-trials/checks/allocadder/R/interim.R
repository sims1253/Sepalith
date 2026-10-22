near <- function(a, b, tol = 1e-8) isTRUE(all.equal(a, b, tolerance = tol))

# spending functions: all spend exactly alpha at t = 1
stopifnot(near(spendObrienFleming(1), 0.05), near(spendPocock(1), 0.05))
stopifnot(near(spendPowerFamily(c(0.5, 1), rho = 2), c(0.0125, 0.05)))
stopifnot(spendObrienFleming(0.5) < spendPocock(0.5))

# equalLooks
stopifnot(near(equalLooks(4), c(0.25, 0.5, 0.75, 1)))

# spendingBoundaries: decreasing OBF bounds; a single look gives the fixed design
sb <- spendingBoundaries(equalLooks(3), nSim = 2e4)
stopifnot(nrow(sb) == 3, !is.unsorted(rev(sb$zBound)), near(sum(sb$alphaInc), 0.05))
one <- spendingBoundaries(1, nSim = 2e4)
stopifnot(nrow(one) == 1, abs(one$zBound - 1.96) < 0.05)
stopifnot(inherits(try(spendingBoundaries(c(0.5, 0.9)), silent = TRUE), "try-error"))

# haybittlePeto
hp <- haybittlePeto(3)
stopifnot(identical(hp$zBound[1:2], c(3, 3)), near(hp$zBound[3], qnorm(0.975)))

# interimDecision
stopifnot(interimDecision(3.5, hp, 1) == "stop_efficacy")
stopifnot(interimDecision(1, hp, 2) == "continue")
stopifnot(interimDecision(-1, hp, 2, futilityZ = 0) == "stop_futility")
stopifnot(interimDecision(1, hp, 3, futilityZ = 0) == "final_not_significant")

# formatBoundaries
fb <- formatBoundaries(sb)
stopifnot(identical(names(fb), c("Look", "Information", "Z boundary", "Nominal p")))
stopifnot(identical(formatBoundaries(hp)$`Z boundary`[1], "3.000"))

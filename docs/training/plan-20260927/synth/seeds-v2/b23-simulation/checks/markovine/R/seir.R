# skipped: needs deSolve: seir.solve(), seir.fit.beta()
st <- c(S = 90, E = 5, I = 5, R = 0)
d <- seir.derivs(0, st, list(beta = 0.5, sigma = 0.2, gamma = 0.1))
stopifnot(isTRUE(all.equal(sum(d[[1]]), 0)), isTRUE(all.equal(d[[1]][1], -2.25)))
stopifnot(d$incidence == 1)
d2 <- seirs.derivs(0, c(S = 0, E = 0, I = 0, R = 10),
  list(beta = 0.5, sigma = 0.2, gamma = 0.1, omega = 0.1))
stopifnot(identical(d2[[1]], c(1, 0, 0, -1)))

stopifnot(identical(seir.initial(100, I0 = 2), c(S = 98, E = 0, I = 2, R = 0)))
stopifnot(inherits(try(seir.initial(5, I0 = 6), silent = TRUE), "try-error"))
stopifnot(isTRUE(all.equal(seir.r0(list(beta = 0.3, gamma = 0.1)), 3)))
stopifnot(seir.r0(c(beta = 1, gamma = 0.5)) == 2)

stopifnot(seir.final.size(0.9) == 0)
z <- seir.final.size(2)
stopifnot(abs(z - (1 - exp(-2 * z))) < 1e-10, z > 0.79, z < 0.8)

traj <- data.frame(time = 0:3, S = c(99, 90, 70, 60), E = c(0, 4, 5, 2),
  I = c(1, 5, 15, 8), R = c(0, 1, 10, 30), incidence = c(0, 3, 6, 2))
sm <- seir.summary(traj)
stopifnot(sm$peak.time == 2)
stopifnot(isTRUE(all.equal(c(sm$peak.prevalence, sm$attack.rate), c(0.15, 0.39))))

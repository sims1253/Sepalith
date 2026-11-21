# skipped: needs deSolve (simulateOde)
err = function(expr) tryCatch({ expr; "" }, error = conditionMessage)
s0 = initState(1000, exposed = 10, infectious = 5)
stopifnot(identical(s0, c(S = 985, E = 10, I = 5, R = 0)))
stopifnot(err(initState(0)) == "`N` must be a single positive number.")
stopifnot(err(initState(10, exposed = -1)) == "`exposed` must be a single non-negative number.")
stopifnot(err(initState(10, infectious = NA_real_ + c(1, 2))) ==
  "`infectious` must be a single non-negative number.")
stopifnot(err(initState(10,
  recovered = -3)) == "`recovered` must be a single non-negative number.")
stopifnot(err(initState(10, 5, 5, 5)) == "Initial compartments exceed the population size `N`.")

step = function(state, params, dt) {
  N = sum(state)
  inf = params$beta * state[["S"]] * state[["I"]] / N * dt
  rec = params$gamma * state[["I"]] * dt
  state + c(S = -inf, E = 0, I = inf - rec, R = rec)
}
set.seed(1)
tr = simulateTrajectory(s0, step, list(beta = 0.5, gamma = 0.1), days = 60, dt = 0.25)
stopifnot(nrow(tr) == 61L, identical(names(tr), c("time", "S", "E", "I", "R")))
stopifnot(tr$time[61] == 60, isTRUE(all.equal(tr$S[1], 985)), all(diff(tr$S) <= 0))
stopifnot(err(simulateTrajectory(s0, step, list(), 5, dt = 0.3)) ==
  "`dt` must divide one day into a whole number of steps.")

sm = trajectorySummary(tr)
stopifnot(identical(names(sm), c("minS", "peakE", "peakI", "finalR")))
stopifnot(sm$minS$time == 60, sm$peakI$value == max(tr$I), sm$finalR$value == tr$R[61])
stopifnot(sm$peakE$time == 0, sm$peakI$time > 0)
stopifnot(err(trajectorySummary(tr[, -2])) == "Trajectory is missing column `S`.")
stopifnot(err(simulateTrajectory(s0, "step", list(), 5)) == "`stepFun` must be a function.")
stopifnot(err(simulateTrajectory(s0, step, list(),
  0)) == "`days` must be a single positive number.")
stopifnot(err(simulateTrajectory(s0, step, list(), 5, dt = -1)) ==
  "`dt` must be a single positive number.")

stopifnot(finalSize(0.9) == 0)
z = finalSize(2)
stopifnot(abs(z - (1 - exp(-2 * z))) < 1e-9, z > 0.79, z < 0.8)
stopifnot(abs(finalSize(3) - 0.9405) < 1e-3)

stopifnot(identical(checkTrajectory(tr), tr))
bad = tr; bad$E = NULL
stopifnot(err(checkTrajectory(bad)) == "Trajectory is missing column `E`.")
bad = tr; bad$time[3] = 1
stopifnot(err(checkTrajectory(bad)) == "Column `time` must be strictly increasing.")
for (col in c("S", "E", "I", "R")) {
  bad = tr; bad[[col]][2] = -1
  stopifnot(err(checkTrajectory(bad)) ==
    sprintf("Column `%s` must be non-negative and complete.", col))
}
bad = tr; bad$R[5] = bad$R[5] + 10
stopifnot(err(checkTrajectory(bad)) == "The population size is not conserved.")

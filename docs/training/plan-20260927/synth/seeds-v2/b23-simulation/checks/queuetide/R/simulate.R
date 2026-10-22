# skipped: needs R6 (ServerPool, EventQueue): handleArrival(),
# handleDeparture(), runQueueSim(), replicateQueueSim()
set.seed(31)
a = arrivalsPoisson(2, 10)
stopifnot(!is.unsorted(a), all(a >= 0 & a <= 10))
s = arrivalsScheduled(1, 5)
stopifnot(identical(s, c(0, 1, 2, 3, 4, 5)))
sj = arrivalsScheduled(2, 5, jitter = 0.2)
stopifnot(!is.unsorted(sj), all(sj >= 0 & sj <= 5))

sim = list(
  customers = data.frame(
    id = 1:4, arrive = c(0, 1, 2, 3), start = c(0, 2, NA, NA),
    finish = c(2, 4, NA, NA), wait = c(0, 1, NA, NA), sojourn = c(2, 3, NA, NA),
    balked = c(FALSE, FALSE, TRUE, FALSE)
  ),
  utilisation = c(0.5, 1),
  endTime = 4
)
sm = summariseQueueSim(sim)
stopifnot(nrow(sm) == 1, sm$served == 2, sm$balkRate == 0.25, sm$meanWait == 0.5)
stopifnot(sm$meanSojourn == 2.5, sm$utilisation == 0.75)

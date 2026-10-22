# skipped: compare_scenarios() needs run_herd() from another package file
pop <- data.frame(
  id = 1:6, x = c(0, 1, 2, 8, 9, 5), y = rep(0, 6),
  state = c("S", "S", "S", "S", "I", "R"), days_in_state = c(0L, 0L, 0L, 0L, 4L, 0L),
  stringsAsFactors = FALSE
)
attr(pop, "arena") <- c(width = 10, height = 10)

set.seed(4)
v <- vaccinate_random(pop, coverage = 0.5)
stopifnot(sum(v$state == "R") == 3, sum(v$state == "S") == 2)
stopifnot(identical(vaccinate_random(pop, coverage = 0), pop))

r <- vaccinate_ring(pop, coverage = 0.5, centre = c(0, 0))
stopifnot(identical(r$state[1:4], c("R", "R", "S", "S")))

q <- quarantine_infectious(pop, delay = 2L, compliance = 1)
stopifnot(q$x[5] == -100, q$y[5] == -100, identical(q$x[1:4], pop$x[1:4]))
stopifnot(identical(apply_intervention(pop, "none"), pop))
stopifnot(sum(apply_intervention(pop, "ring", coverage = 1)$state == "R") == 5)

counts <- data.frame(
  day = 0:4, S = c(9L, 8L, 6L, 5L, 5L), E = c(0L, 1L, 1L, 0L, 0L),
  I = c(1L, 1L, 2L, 1L, 0L), R = c(0L, 0L, 1L, 4L, 5L)
)
stopifnot(identical(peak_infections(counts), list(day = 2L, size = 2L)))
stopifnot(all.equal(attack_rate(counts), 0.4))
stopifnot(outbreak_duration(counts) == 3L)
counts$E <- 0L
counts$I <- 0L
stopifnot(is.na(outbreak_duration(counts)))

# skipped: step_population() and run_herd() need move_random() and
# validate_population() from other package files
dist <- c(0, 1, 2, 3)
stopifnot(all.equal(kernel_constant(dist, 0.3, 2), c(0.3, 0.3, 0.3, 0)))
stopifnot(kernel_exponential(0, 0.4, 2) == 0.4, kernel_exponential(3, 0.4, 2) == 0)
stopifnot(kernel_gaussian(0, 0.4, 2) == 0.4, all(diff(kernel_gaussian(dist[1:3], 1, 2)) < 0))

pop <- data.frame(
  id = 1:4, x = c(0, 1, 50, 2), y = c(0, 0, 50, 0),
  state = c("I", "S", "S", "R"), days_in_state = c(3L, 3L, 3L, 3L),
  stringsAsFactors = FALSE
)
set.seed(2)
out <- expose_contacts(pop, beta = 1, radius = 2)
stopifnot(identical(out$state, c("I", "E", "S", "R")), out$days_in_state[2] == 0L)
none <- pop
none$state[1] <- "R"
stopifnot(identical(expose_contacts(none, beta = 1), none))

pe <- pop
pe$state[2:3] <- "E"
pe <- progress_exposed(pe, rate = 1)
stopifnot(identical(pe$state, c("I", "I", "I", "R")), all(pe$days_in_state[2:3] == 0L))
stopifnot(identical(progress_exposed(pop, rate = 1), pop))

pr <- progress_infectious(pop, rate = 1)
stopifnot(identical(pr$state, c("R", "S", "S", "R")), pr$days_in_state[1] == 0L)
stopifnot(identical(progress_infectious(pop, rate = 0)$state, pop$state))

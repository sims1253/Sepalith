test_that("kernels are zero beyond the contact radius", {
  dist <- c(0, 1, 2, 2.5, 10)
  expect_equal(kernel_constant(dist, 0.3, 2), c(0.3, 0.3, 0.3, 0, 0))
  expect_equal(kernel_exponential(dist, 0.3, 2)[4:5], c(0, 0))
  expect_equal(kernel_gaussian(dist, 0.3, 2)[4:5], c(0, 0))
})

test_that("kernels peak at distance zero", {
  dist <- seq(0, 2, by = 0.25)
  expo <- kernel_exponential(dist, 0.5, 2)
  gaus <- kernel_gaussian(dist, 0.5, 2)
  expect_equal(expo[1], 0.5)
  expect_equal(gaus[1], 0.5)
  expect_true(all(diff(expo) <= 0))
  expect_true(all(diff(gaus) <= 0))
})

test_that("expose_contacts leaves a population without infectious agents alone", {
  set.seed(11)
  pop <- new_population(50, n_infected = 0)
  out <- expose_contacts(pop, beta = 1, radius = 100)
  expect_identical(out$state, pop$state)
  expect_identical(out$days_in_state, pop$days_in_state)
})

test_that("expose_contacts with beta one infects every susceptible in range", {
  set.seed(12)
  pop <- new_population(30, n_infected = 1)
  out <- expose_contacts(pop, beta = 1, radius = 1000, kernel = "constant")
  expect_equal(sum(out$state == "S"), 0)
  expect_equal(sum(out$state == "E"), 29)
  expect_equal(sum(out$state == "I"), 1)
})

test_that("progress_exposed moves every exposed agent when rate is one", {
  pop <- new_population(10, n_infected = 0)
  pop$state[1:4] <- "E"
  pop$days_in_state[1:4] <- 3L
  out <- progress_exposed(pop, rate = 1)
  expect_equal(out$state[1:4], rep("I", 4))
  expect_equal(out$days_in_state[1:4], rep(0L, 4))
  expect_equal(out$state[5:10], rep("S", 6))
})

test_that("progress_infectious moves every infectious agent when rate is one", {
  pop <- new_population(10, n_infected = 0)
  pop$state[1:4] <- "I"
  pop$days_in_state[1:4] <- 3L
  out <- progress_infectious(pop, rate = 1)
  expect_equal(out$state[1:4], rep("R", 4))
  expect_equal(out$days_in_state[1:4], rep(0L, 4))
  expect_equal(out$state[5:10], rep("S", 6))
})

test_that("run_herd conserves the population size", {
  set.seed(13)
  pop <- new_population(80, n_infected = 2)
  run <- run_herd(pop, days = 30, params = list(beta = 0.4, radius = 5))
  totals <- run$counts$S + run$counts$E + run$counts$I + run$counts$R
  expect_true(all(totals == 80))
  expect_equal(run$counts$day[1], 0L)
  expect_null(run$snapshots)
})

test_that("run_herd stops early once the outbreak is over", {
  set.seed(14)
  pop <- new_population(40, n_infected = 1)
  run <- run_herd(pop, days = 200, params = list(beta = 0, gamma = 1),
    keep_snapshots = TRUE)
  expect_lt(nrow(run$counts), 201)
  expect_equal(tail(run$counts$I, 1), 0L)
  expect_length(run$snapshots, nrow(run$counts))
})

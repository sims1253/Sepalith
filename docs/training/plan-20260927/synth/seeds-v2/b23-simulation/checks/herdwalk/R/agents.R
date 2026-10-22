set.seed(1)
pop <- new_population(20, n_infected = 3, width = 10, height = 5)
stopifnot(inherits(pop, "herd_population"), nrow(pop) == 20)
stopifnot(sum(pop$state == "I") == 3, all(pop$x <= 10), all(pop$y <= 5))

stopifnot(identical(validate_population(pop), pop))
bad <- pop
bad$state[1] <- "Z"
stopifnot(inherits(try(validate_population(bad), silent = TRUE), "try-error"))

stopifnot(population_size(pop) == 20L)
counts <- population_counts(pop)
stopifnot(identical(names(counts), c("S", "E", "I", "R")), sum(counts) == 20L)
stopifnot(counts[["E"]] == 0L)

more <- add_agents(pop, 5, state = "E")
stopifnot(population_size(more) == 25L, identical(more$id, 1:25))
stopifnot(inherits(more, "herd_population"), !is.null(attr(more, "arena")))
stopifnot(identical(add_agents(pop, 0), pop))

fewer <- remove_agents(pop, c(1, 2, 99))
stopifnot(population_size(fewer) == 18L, !any(c(1, 2) %in% fewer$id))
stopifnot(identical(remove_agents(pop, integer(0)), pop))

pop2 <- pop
pop2$x[1:3] <- c(0, 1, 5)
pop2$y[1:3] <- c(0, 0, 5)
nb <- agent_neighbours(pop2[1:3, ], 1, radius = 2)
stopifnot(identical(nb, 2L))
stopifnot(inherits(try(agent_neighbours(pop, 999), silent = TRUE), "try-error"))

out <- capture.output(res <- print(pop))
stopifnot(identical(res, pop), grepl("20 agents", out[1]), length(out) == 5)

sm <- summary(pop)
stopifnot(nrow(sm) == 4, isTRUE(all.equal(sum(sm$prop), 1)))
stopifnot(is.na(sm$mean_days[sm$state == "E"]))
empty <- remove_agents(pop, pop$id)
stopifnot(all(is.na(summary(empty)$prop)))

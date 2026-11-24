s <- new_sim_state(2, 5, 1.5)
stopifnot(inherits(s, "sim_state"), s$n_servers == 2L, s$capacity == 5L)
stopifnot(nrow(s$queue) == 0L, all(s$counts == 0L))
err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
stopifnot(err(new_sim_state("a", 5, 1)) == "`n_servers` must be a single number.")
stopifnot(err(new_sim_state(0, 5, 1)) == "`n_servers` must be greater than 0.")
stopifnot(err(new_sim_state(1, NA, 1)) == "`capacity` must be a single number.")
stopifnot(err(new_sim_state(1, 0, 1)) == "`capacity` must be greater than 0.")
stopifnot(err(new_sim_state(1, 2, c(1, 2))) == "`service_rate` must be a single number.")
stopifnot(err(new_sim_state(1, 2, -1)) == "`service_rate` must be greater than 0.")

stopifnot(identical(validate_state(s), s))
bad <- s; bad$time <- NA
stopifnot(err(validate_state(bad)) == "State field `time` is malformed.")
bad <- s; bad$busy <- 3L
stopifnot(err(validate_state(bad)) == "State field `busy` is malformed.")
bad <- s; bad$waiting <- c(1L, 1L)
stopifnot(err(validate_state(bad)) == "State field `waiting` is malformed.")
bad <- s; bad$queue <- data.frame(time = c(2, 1))
stopifnot(err(validate_state(bad)) == "State field `queue` is malformed.")
bad <- s; bad$counts[["balk"]] <- -1L
stopifnot(err(validate_state(bad)) == "State field `counts` is malformed.")
bad <- s; bad$counts <- NULL
stopifnot(err(validate_state(bad)) == "State is missing field `counts`.")

q <- s
q$queue <- data.frame(time = c(1, 1, 3), type = c("arrival", "balk", "arrival"),
  entity = c(2L, 3L, 1L))
p <- pop_event(q)
stopifnot(p$event$entity == 2L, nrow(p$state$queue) == 2L)
stopifnot(is.null(pop_event(s)$event))

snap <- state_snapshot(q)
stopifnot(identical(names(snap), c("time", "busy", "waiting", "scheduled", "served", "balked")))
stopifnot(snap$scheduled == 3L, snap$served == 0L, snap$balked == 0L)
stopifnot(identical(state_snapshot(q, c("busy", "waiting")), list(busy = 0L, waiting = 0L)))

d <- drop_events(q, "arrival")
stopifnot(nrow(d$queue) == 1L, d$queue$type == "balk")
d2 <- drop_events(q, "arrival", entity = 1L)
stopifnot(nrow(d2$queue) == 2L, identical(d2$queue$entity, c(2L, 3L)))
stopifnot(nrow(drop_events(q, "arrival", after = 2)$queue) == 2L)
stopifnot(nrow(drop_events(q, "arrival", before = 2)$queue) == 2L)
stopifnot(nrow(drop_events(q, "arrival", after = 0.5, before = 0.9)$queue) == 3L)
stopifnot(err(drop_events(q, "arrival", after = "a")) == "`after` must be a single number.")
stopifnot(err(drop_events(q, "arrival", before = 1:2)) == "`before` must be a single number.")

lst <- list(time = 10, start_time = 0, area_busy = 15, area_queue = 5,
  sojourn = c("1" = 3, "2" = 5, "3" = 0), balked = 3L)
lc <- little_check(lst)
stopifnot(identical(names(lc), c("L", "lambda", "W", "gap")))
stopifnot(lc$L == 2, lc$lambda == 0.2, lc$W == 4, lc$gap == 0.6)

# skipped: needs new_sim_state() and the handlers from state.R and handlers.R
# (run_queue_sim, compare_servers)
set.seed(1)
for (p in c("poisson", "uniform", "fixed", "batch")) {
  a <- generate_arrivals(7, p, rate = 2)
  stopifnot(nrow(a) == 7L, !is.unsorted(a$time), all(a$type == "arrival"))
}
stopifnot(isTRUE(all.equal(generate_arrivals(4, "fixed", rate = 2)$time, c(0.5, 1, 1.5, 2))))
b <- generate_arrivals(7, "batch", batch_size = 3L)
stopifnot(length(unique(b$time)) == 3L, b$time[1] == b$time[3])
u <- generate_arrivals(200, "uniform", rate = 4)
stopifnot(all(diff(c(0, u$time)) <= 0.5))
stopifnot(inherits(try(generate_arrivals(0), silent = TRUE), "try-error"))

st <- list(time = 10, start_time = 0, n_servers = 2L, area_busy = 15, area_queue = 5,
  waited = c("1" = 0, "2" = 2), sojourn = c("1" = 3, "2" = 5, "3" = 0),
  balked = 3L, counts = c(arrival = 4L, service_start = 2L, departure = 2L, balk = 1L),
  log = list(
    data.frame(time = 0.5, type = "arrival", entity = 1L, busy = 0L, waiting = 1L),
    data.frame(time = 1.5, type = "balk", entity = 3L, busy = 1L, waiting = 0L),
    data.frame(time = 2.5, type = "arrival", entity = 2L, busy = 1L, waiting = 1L)))
m <- queue_metrics(st)
stopifnot(identical(names(m), c("utilisation", "mean_queue", "mean_wait",
  "mean_sojourn", "balk_rate")))
stopifnot(isTRUE(all.equal(unname(m), c(0.75, 0.5, 1, 8 / 3, 0.25))))
stopifnot(identical(names(queue_metrics(st, "mean_wait")), "mean_wait"))
st0 <- st; st0$time <- 0
stopifnot(inherits(try(queue_metrics(st0), silent = TRUE), "try-error"))

tr <- trace_to_frame(st)
stopifnot(nrow(tr) == 3L, identical(names(tr), c("time", "type", "entity", "busy", "waiting")))
stopifnot(nrow(trace_to_frame(list(log = list()))) == 0L)

ce <- count_events(tr, width = 2)
stopifnot(identical(ce$window, c(0, 2)), identical(ce$arrival, c(1L, 1L)))
stopifnot(identical(ce$balk, c(1L, 0L)), identical(names(ce),
  c("window", "arrival", "service_start", "departure", "balk")))

h <- list(arrival = identity, balk = identity)
h2 <- override_handlers(h, balk = function(state, event) state)
stopifnot(!identical(h2$balk, h$balk), identical(h2$arrival, h$arrival))
stopifnot(conditionMessage(tryCatch(override_handlers(h, renege = identity),
  error = identity)) == "No handler registered for event type `renege`.")
stopifnot(conditionMessage(tryCatch(override_handlers(h, balk = 1),
  error = identity)) == "Handler for event type `balk` must be a function.")
ce2 <- count_events(tr, width = 1, from = 1, to = 2)
stopifnot(identical(ce2$window, 1), ce2$balk == 1L, ce2$arrival == 0L)
stopifnot(sum(count_events(tr, to = 1)$arrival) == 1L)

ev <- function(t, type, id) list(time = t, type = type, entity = id)
hd <- list(arrival = function(state, event) {
  state$time <- event$time
  state$seen <- c(state$seen, event$entity)
  state
})
d <- dispatch_event(list(time = 0), ev(1, "arrival", 4L), hd)
stopifnot(d$time == 1, d$seen == 4L)
msg <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
stopifnot(msg(dispatch_event(list(time = 0), ev(1, "balk", 1L), hd)) ==
  "No handler registered for event type `balk`.")
stopifnot(msg(dispatch_event(list(time = 0), ev(NA, "arrival", 1L), hd)) ==
  "Event field `time` is missing.")
stopifnot(msg(dispatch_event(list(time = 0), list(time = 1, entity = 1L), hd)) ==
  "Event field `type` is missing.")
stopifnot(msg(dispatch_event(list(time = 0), ev(1, "arrival", NA), hd)) ==
  "Event field `entity` is missing.")
stopifnot(grepl("lies before the clock", msg(dispatch_event(list(time = 5), ev(1,
  "arrival", 1L), hd))))
stopifnot(msg(dispatch_event(list(time = 0), ev(1, "arrival", 1L), list(identity))) ==
  "`handlers` must be a named list of functions.")

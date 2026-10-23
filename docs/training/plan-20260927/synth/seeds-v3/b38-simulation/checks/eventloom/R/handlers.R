mk_state <- function(n_servers = 1L, capacity = 2L, service_rate = 1) {
  list(time = 0, start_time = 0, n_servers = n_servers, capacity = capacity,
    service_rate = service_rate, busy = 0L, waiting = integer(),
    queue = data.frame(time = numeric(), type = character(), entity = integer()),
    counts = c(arrival = 0L, service_start = 0L, departure = 0L, balk = 0L),
    arrived_at = numeric(), waited = numeric(), sojourn = numeric(),
    balked = integer(), area_queue = 0, area_busy = 0, log = list())
}
ev <- function(t, type, id) list(time = t, type = type, entity = as.integer(id))
err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
set.seed(42)
s <- mk_state(service_rate = 1e-3)

stopifnot(identical(new_event(2, "arrival", 3), data.frame(time = 2, type = "arrival",
  entity = 3L)))
stopifnot(err(new_event(NA, "arrival", 1)) ==
  "Event field `time` must be a single non-missing value.")
stopifnot(err(new_event(1, 2, 1)) == "Event field `type` must be a single non-missing value.")
stopifnot(err(new_event(1, "balk", c(1, 2))) ==
  "Event field `entity` must be a single non-missing value.")

q <- schedule_event(s, 3, "arrival", 1)
q <- schedule_event(q, 1, "arrival", 2)
q <- schedule_event(q, 1, "balk", 3)
stopifnot(identical(q$queue$time, c(1, 1, 3)), identical(q$queue$entity, c(2L, 3L, 1L)))

s1 <- handle_arrival(s, ev(1, "arrival", 1))
stopifnot(s1$time == 1, s1$counts[["arrival"]] == 1L, identical(s1$waiting, 1L))
stopifnot(s1$queue$type == "service_start", length(s1$log) == 1L)

s2 <- handle_service_start(s1, ev(1, "service_start", 1))
stopifnot(s2$busy == 1L, length(s2$waiting) == 0L, s2$waited[["1"]] == 0)
stopifnot("departure" %in% s2$queue$type, s2$counts[["service_start"]] == 1L)
stopifnot(identical(handle_service_start(s2, ev(1, "service_start", 1))$counts,
  s2$counts))

s3 <- handle_arrival(s2, ev(1.5, "arrival", 2))
s3 <- handle_arrival(s3, ev(1.6, "arrival", 3))
s3 <- handle_arrival(s3, ev(1.7, "arrival", 4))
stopifnot(identical(s3$waiting, 2:3), "balk" %in% s3$queue$type)

s4 <- handle_balk(s3, ev(1.7, "balk", 4))
stopifnot(s4$counts[["balk"]] == 1L, identical(s4$balked, 4L), s4$sojourn[["4"]] == 0)

dep <- s4$queue[s4$queue$type == "departure", ]
s5 <- handle_departure(s4, ev(dep$time, "departure", 1))
stopifnot(s5$busy == 0L, s5$counts[["departure"]] == 1L)
stopifnot(abs(s5$sojourn[["1"]] - (dep$time - 1)) < 1e-9)
stopifnot(any(s5$queue$type == "service_start" & s5$queue$entity == 2L))

h <- event_handlers()
stopifnot(identical(names(h), c("arrival", "service_start", "departure", "balk")))
stopifnot(all(vapply(h, is.function, logical(1))))

b <- s; b$busy <- 1L; b$waiting <- c(4L, 5L)
b <- handle_balk(b, ev(2, "balk", 9))
stopifnot(b$area_queue == 4, b$area_busy == 2, b$log[[1]]$waiting == 2L)

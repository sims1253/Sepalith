#' Arrival time generators
#'
#' `arrivalsPoisson()` generates arrival times of a homogeneous Poisson
#' process on `[0, horizon]`.
#'
#' @param rate Mean number of arrivals per unit time.
#' @param horizon End of the simulated time window.
#' @return A sorted numeric vector of arrival times.
#' @export
arrivalsPoisson = function(rate, horizon) {
  stopifnot(rate > 0, horizon > 0)
  # Given the count, Poisson arrival times are iid uniform on the window.
  n = stats::rpois(1, rate * horizon)
  sort(stats::runif(n, 0, horizon))
}

#' @description `arrivalsScheduled()` generates appointment-style arrivals
#'   at a fixed spacing, each jittered by a uniform punctuality error.
#' @param jitter Maximum absolute deviation from the scheduled time.
#' @rdname arrivalsPoisson
#' @export
arrivalsScheduled = function(rate, horizon, jitter = 0) {
  stopifnot(rate > 0, horizon > 0, jitter >= 0)
  slots = seq(0, horizon, by = 1 / rate)
  times = slots + stats::runif(length(slots), -jitter, jitter)
  # Early or late customers can fall outside the window; drop them.
  sort(times[times >= 0 & times <= horizon])
}

#' Handle an arrival event
#'
#' Starts service immediately if a server is free, otherwise places the
#' customer in line or records a balk when the line is full.
#'
#' @param event A `qt_event` of type `"arrival"`.
#' @param pool A [ServerPool] object.
#' @param queue An [EventQueue] object holding future events.
#' @param log An environment collecting per-customer records.
#' @param sampler A service time sampler, see [makeServiceSampler()].
#' @return `NULL`, invisibly; called for its side effects.
#' @keywords internal
handleArrival = function(event, pool, queue, log, sampler) {
  id = event$payload$id
  now = event$time
  log$arrive[id] = now

  server = pool$freeServer()
  if (!is.na(server)) {
    pool$startService(server, id, now)
    log$start[id] = now
    queue$push(newEvent(now + sampler(1), "departure", list(id = id, server = server)))
  } else if (!pool$enqueue(id, now)) {
    # Line is full: the customer leaves without being served.
    log$balked[id] = TRUE
  }
  invisible(NULL)
}

#' Handle a departure event
#'
#' Frees the server and, if anyone is waiting, starts serving the next
#' customer in line on the same server.
#'
#' @param event A `qt_event` of type `"departure"`.
#' @inheritParams handleArrival
#' @keywords internal
handleDeparture = function(event, pool, queue, log, sampler) {
  server = event$payload$server
  now = event$time
  done = pool$endService(server, now)
  log$finish[done] = now

  # The freed server immediately picks up the head of the line.
  nextUp = pool$dequeue()
  if (!is.null(nextUp)) {
    pool$startService(server, nextUp$id, now)
    log$start[nextUp$id] = now
    queue$push(newEvent(now + sampler(1), "departure", list(id = nextUp$id, server = server)))
  }
  invisible(NULL)
}

#' Run a discrete-event queue simulation
#'
#' Simulates a multi-server queue fed by a vector of arrival times, processing
#' events in time order until the queue is empty or the horizon is reached.
#'
#' @param arrivals Numeric vector of arrival times.
#' @param servers Number of parallel servers.
#' @param sampler A service time sampler, see [makeServiceSampler()].
#' @param horizon Time after which no further events are processed.
#' @param queueLimit Maximum number of waiting customers.
#' @return A list with elements `customers` (a data frame with one row per
#'   customer), `utilisation` (per-server busy fractions) and `endTime`.
#' @export
runQueueSim = function(arrivals, servers = 1L, sampler = serviceExponential(1),
                       horizon = Inf, queueLimit = Inf) {
  n = length(arrivals)

  pool = ServerPool$new(capacity = servers, queueLimit = queueLimit)
  queue = eventQueueFromTimes(arrivals, type = "arrival")
  # Per-customer records live in an environment so the handlers can
  # update them in place.
  log = new.env(parent = emptyenv())
  log$arrive = rep(NA_real_, n)
  log$start = rep(NA_real_, n)
  log$finish = rep(NA_real_, n)
  log$balked = rep(FALSE, n)

  now = 0
  while (!queue$isEmpty()) {
    event = queue$pop()
    if (event$time > horizon) {
      break
    }
    now = event$time
    switch(event$type,
      arrival = handleArrival(event, pool, queue, log, sampler),
      departure = handleDeparture(event, pool, queue, log, sampler),
      stop("Unknown event type: ", event$type, call. = FALSE)
    )
  }

  customers = data.frame(
    id = seq_len(n),
    arrive = log$arrive,
    start = log$start,
    finish = log$finish,
    wait = log$start - log$arrive,
    sojourn = log$finish - log$arrive,
    balked = log$balked
  )
  list(
    customers = customers,
    utilisation = pool$utilisation(now),
    endTime = now
  )
}

#' Summarise a queue simulation
#'
#' Reduces the per-customer records of a run to headline performance
#' measures.
#'
#' @param sim A list as returned by [runQueueSim()].
#' @return A one-row data frame of summary measures.
#' @export
summariseQueueSim = function(sim) {
  cust = sim$customers
  served = !is.na(cust$finish)

  data.frame(
    customers = nrow(cust),
    served = sum(served),
    balkRate = mean(cust$balked),
    meanWait = mean(cust$wait[served]),
    p90Wait = unname(stats::quantile(cust$wait[served], 0.9, na.rm = TRUE)),
    meanSojourn = mean(cust$sojourn[served]),
    utilisation = mean(sim$utilisation),
    endTime = sim$endTime
  )
}

#' Replicate a queue simulation
#'
#' Runs independent replications of a queue with Poisson arrivals and
#' stacks their summaries, adding normal-theory confidence intervals for
#' the mean waiting time.
#'
#' @param reps Number of replications.
#' @param rate Mean number of arrivals per unit time.
#' @param horizon End of the simulated time window.
#' @param servers Number of parallel servers.
#' @param sampler A service time sampler, see [makeServiceSampler()].
#' @param level Confidence level of the interval.
#' @return A list with elements `runs` (one row per replication) and
#'   `meanWait` (estimate with lower and upper bounds).
#' @export
replicateQueueSim = function(reps, rate, horizon, servers = 1L,
                             sampler = serviceExponential(1), level = 0.95) {
  stopifnot(reps >= 2)

  runs = vector("list", reps)
  for (r in seq_len(reps)) {
    arrivals = arrivalsPoisson(rate, horizon)
    sim = runQueueSim(arrivals, servers = servers, sampler = sampler)
    runs[[r]] = cbind(rep = r, summariseQueueSim(sim))
  }
  runs = do.call(rbind, runs)

  # Runs in which nobody was served have no mean wait; leave them out of
  # both the estimate and the standard error.
  ok = !is.na(runs$meanWait)
  est = mean(runs$meanWait[ok])
  se = stats::sd(runs$meanWait[ok]) / sqrt(sum(ok))
  z = stats::qnorm(1 - (1 - level) / 2)
  list(
    runs = runs,
    meanWait = c(estimate = est, lower = est - z * se, upper = est + z * se)
  )
}

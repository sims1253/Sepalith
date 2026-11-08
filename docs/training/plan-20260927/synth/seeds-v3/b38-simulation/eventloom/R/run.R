#' Generate arrival events
#'
#' @param n Number of arrivals.
#' @param process Arrival process: `"poisson"` (exponential gaps),
#'   `"uniform"` (uniform gaps with the same mean), `"fixed"` (constant
#'   gaps) or `"batch"` (groups of `batch_size` at Poisson times).
#' @param rate Mean number of arrivals per unit time.
#' @param batch_size Group size for the `"batch"` process.
#' @param start Time of the clock before the first arrival.
#' @return A data frame with columns `time`, `type` and `entity`.
#' @export
generate_arrivals <- function(n, process = c("poisson", "uniform", "fixed", "batch"),
                              rate = 1, batch_size = 3L, start = 0) {
  process <- match.arg(process)
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop("`n` must be a single positive number.", call. = FALSE)
  }
  if (!is.numeric(rate) || length(rate) != 1L || rate <= 0) {
    stop("`rate` must be a single positive number.", call. = FALSE)
  }

  times <- switch(process,
    poisson = {
      gaps <- stats::rexp(n, rate = rate)
      start + cumsum(gaps)
    },
    uniform = {
      gaps <- stats::runif(n, min = 0, max = 2 / rate)
      start + cumsum(gaps)
    },
    fixed = {
      gaps <- rep(1 / rate, n)
      start + cumsum(gaps)
    },
    batch = {
      gaps <- stats::rexp(ceiling(n / batch_size), rate = rate / batch_size)
      start + rep(cumsum(gaps), each = batch_size)[seq_len(n)]
    }
  )
  arrivals <- data.frame(time = times, type = "arrival", entity = seq_len(n))
  return(arrivals)
}

#' Run a queue simulation
#'
#' @param arrivals A data frame of arrival events, see [generate_arrivals()].
#' @param n_servers Number of parallel servers.
#' @param capacity Maximum number of entities allowed in the waiting line.
#' @param service_rate Rate of the exponential service time distribution.
#' @param max_events Stop after this many events have been handled.
#' @param handlers A named list of handlers, see [event_handlers()].
#' @return The final `sim_state`, with an extra element `n_events`.
#' @export
run_queue_sim <- function(arrivals, n_servers = 1L, capacity = 10L, service_rate = 1,
                          max_events = Inf, handlers = event_handlers()) {
  required <- c("time", "type", "entity")
  absent <- setdiff(required, names(arrivals))
  if (length(absent) > 0L) {
    stop(sprintf("`arrivals` is missing column `%s`.", absent[[1L]]), call. = FALSE)
  }
  state <- new_sim_state(n_servers, capacity, service_rate, start_time = 0)
  queue <- rbind(state$queue, arrivals[, required])
  state$queue <- queue[order(queue$time), , drop = FALSE]

  # Main event loop
  n_done <- 0L
  while (nrow(state$queue) > 0L && n_done < max_events) {
    popped <- pop_event(state)
    state <- dispatch_event(popped$state, popped$event, handlers)
    n_done <- n_done + 1L
  }
  state$n_events <- n_done
  validate_state(state)
  return(state)
}

#' Performance metrics of a finished run
#'
#' @param state A finished `sim_state`, see [run_queue_sim()].
#' @param metrics Metrics to compute, a subset of `"utilisation"`,
#'   `"mean_queue"`, `"mean_wait"`, `"mean_sojourn"` and `"balk_rate"`.
#' @return A named numeric vector with one element per requested metric.
#' @export
queue_metrics <- function(state, metrics = c("utilisation", "mean_queue", "mean_wait",
                                             "mean_sojourn", "balk_rate")) {
  metrics <- match.arg(metrics, several.ok = TRUE)
  horizon <- state$time - state$start_time
  if (horizon <= 0) {
    stop("The run has zero length; no metrics can be computed.", call. = FALSE)
  }
  out <- numeric(0)

  # Metric `utilisation`
  if ("utilisation" %in% metrics) {
    out[["utilisation"]] <- state$area_busy / (horizon * state$n_servers)
  }
  # Metric `mean_queue`
  if ("mean_queue" %in% metrics) {
    out[["mean_queue"]] <- state$area_queue / horizon
  }
  # Metric `mean_wait`
  if ("mean_wait" %in% metrics) {
    out[["mean_wait"]] <- if (length(state$waited)) mean(state$waited) else NA_real_
  }
  # Metric `mean_sojourn`
  if ("mean_sojourn" %in% metrics) {
    out[["mean_sojourn"]] <- if (length(state$sojourn)) mean(state$sojourn) else NA_real_
  }
  # Metric `balk_rate`
  if ("balk_rate" %in% metrics) {
    out[["balk_rate"]] <- state$counts[["balk"]] / max(state$counts[["arrival"]], 1L)
  }
  return(out)
}

#' Convert the event log to a data frame
#'
#' @param state A finished `sim_state`, see [run_queue_sim()].
#' @return A data frame with columns `time`, `type`, `entity`, `busy` and
#'   `waiting`, one row per logged event.
#' @export
trace_to_frame <- function(state) {
  if (length(state$log) == 0L) {
    return(data.frame(time = numeric(), type = character(), entity = integer(),
      busy = integer(), waiting = integer()))
  }
  trace <- do.call(rbind, state$log)
  rownames(trace) <- NULL
  return(trace)
}

#' Count logged events per type and time window
#'
#' @param trace A data frame as returned by [trace_to_frame()].
#' @param width Width of the time windows.
#' @param from,to Optional limits of the counted period; default to the
#'   first and last logged event.
#' @param types Event types to count; one column per type.
#' @return A data frame with column `window` (window start) followed by one
#'   count column per element of `types`.
#' @export
count_events <- function(trace, width = 1, from = NULL, to = NULL,
                         types = c("arrival", "service_start", "departure", "balk")) {
  if (!is.numeric(width) || length(width) != 1L || width <= 0) {
    stop("`width` must be a single positive number.", call. = FALSE)
  }
  # Lower limit of the period
  if (is.null(from)) {
    from <- min(trace$time)
  }
  trace <- trace[trace$time >= from, , drop = FALSE]
  # Upper limit of the period
  if (is.null(to)) {
    to <- max(trace$time)
  }
  trace <- trace[trace$time <= to, , drop = FALSE]

  window <- floor(trace$time / width) * width
  starts <- sort(unique(window))
  out <- data.frame(window = starts)
  for (type in types) {
    hits <- window[trace$type == type]
    out[[type]] <- as.integer(table(factor(hits, levels = starts)))
  }
  return(out)
}

#' Compare server counts on the same arrivals
#'
#' Reruns the queue with each number of servers and the same random seed,
#' so that differences come from the configuration rather than the noise.
#'
#' @param arrivals A data frame of arrival events, see [generate_arrivals()].
#' @param servers Integer vector of server counts to try.
#' @param seed Seed set before each run.
#' @param ... Passed to [run_queue_sim()].
#' @return A data frame with column `n_servers` followed by the columns of
#'   [queue_metrics()].
#' @export
compare_servers <- function(arrivals, servers = 1:3, seed = 1L, ...) {
  if (!is.numeric(servers) || length(servers) == 0L || any(servers < 1)) {
    stop("`servers` must be a vector of positive numbers.", call. = FALSE)
  }
  rows <- vector("list", length(servers))
  for (i in seq_along(servers)) {
    set.seed(seed)
    state <- run_queue_sim(arrivals, n_servers = servers[[i]], ...)
    metrics <- queue_metrics(state)
    rows[[i]] <- data.frame(n_servers = servers[[i]], as.list(metrics))
  }
  out <- do.call(rbind, rows)
  return(out)
}

#' Replace handlers in a dispatch table
#'
#' @param handlers A named list of handlers, see [event_handlers()].
#' @param ... Named handler functions; each name must be an existing event
#'   type in `handlers`.
#' @return The modified dispatch table.
#' @export
override_handlers <- function(handlers, ...) {
  replacements <- list(...)
  unknown <- setdiff(names(replacements), names(handlers))
  if (length(unknown) > 0L) {
    stop(sprintf("No handler registered for event type `%s`.", unknown[[1L]]), call. = FALSE)
  }
  for (type in names(replacements)) {
    if (!is.function(replacements[[type]])) {
      stop(sprintf("Handler for event type `%s` must be a function.", type), call. = FALSE)
    }
    handlers[[type]] <- replacements[[type]]
  }
  return(handlers)
}

#' Dispatch one event to its handler
#'
#' Looks the event type up in a dispatch table, so that alternative models
#' can replace single handlers without touching the run loop.
#'
#' @param state A `sim_state` list as returned by [new_sim_state()].
#' @param event An event list with `time`, `type` and `entity`.
#' @param handlers A named list of handlers, see [event_handlers()].
#' @return The updated `state`.
#' @export
dispatch_event <- function(state, event, handlers = event_handlers()) {
  if (!is.list(handlers) || is.null(names(handlers))) {
    stop("`handlers` must be a named list of functions.", call. = FALSE)
  }
  # Check `time`
  if (is.null(event$time) || is.na(event$time)) {
    stop(sprintf("Event field `%s` is missing.", "time"), call. = FALSE)
  }
  # Check `type`
  if (is.null(event$type) || is.na(event$type)) {
    stop(sprintf("Event field `%s` is missing.", "type"), call. = FALSE)
  }
  # Check `entity`
  if (is.null(event$entity) || is.na(event$entity)) {
    stop(sprintf("Event field `%s` is missing.", "entity"), call. = FALSE)
  }

  handler <- handlers[[event$type]]
  if (is.null(handler)) {
    stop(sprintf("No handler registered for event type `%s`.", event$type), call. = FALSE)
  }
  if (event$time < state$time) {
    stop(sprintf("Event `%s` at %g lies before the clock (%g).", event$type, event$time,
      state$time), call. = FALSE)
  }
  state <- handler(state, event)
  return(state)
}

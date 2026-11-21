#' Create a queue simulation state
#'
#' Builds the mutable state list that the event handlers update. The state
#' starts empty: no events are scheduled and no server is busy.
#'
#' @param n_servers Number of parallel servers.
#' @param capacity Maximum number of entities allowed in the waiting line.
#' @param service_rate Rate of the exponential service time distribution.
#' @param start_time Simulation clock at the start of the run.
#' @return A list of class `sim_state` with elements `time`, `start_time`,
#'   `n_servers`, `capacity`, `service_rate`, `busy`, `waiting`, `queue`,
#'   `counts`, `arrived_at`, `waited`, `sojourn`, `balked`, `area_queue`,
#'   `area_busy` and `log`.
#' @export
new_sim_state <- function(n_servers, capacity, service_rate, start_time = 0) {
  minimum <- c(n_servers = 0, capacity = 0, service_rate = 0)

  # Validate `n_servers`
  if (!is.numeric(n_servers) || length(n_servers) != 1L || is.na(n_servers)) {
    stop(sprintf("`%s` must be a single number.", "n_servers"), call. = FALSE)
  }
  if (n_servers <= minimum[["n_servers"]]) {
    stop(sprintf("`%s` must be greater than %g.", "n_servers", minimum[["n_servers"]]),
      call. = FALSE)
  }
  # Validate `capacity`
  if (!is.numeric(capacity) || length(capacity) != 1L || is.na(capacity)) {
    stop(sprintf("`%s` must be a single number.", "capacity"), call. = FALSE)
  }
  if (capacity <= minimum[["capacity"]]) {
    stop(sprintf("`%s` must be greater than %g.", "capacity", minimum[["capacity"]]),
      call. = FALSE)
  }
  # Validate `service_rate`
  if (!is.numeric(service_rate) || length(service_rate) != 1L || is.na(service_rate)) {
    stop(sprintf("`%s` must be a single number.", "service_rate"), call. = FALSE)
  }
  if (service_rate <= minimum[["service_rate"]]) {
    stop(sprintf("`%s` must be greater than %g.", "service_rate", minimum[["service_rate"]]),
      call. = FALSE)
  }

  state <- list(
    time = start_time,
    start_time = start_time,
    n_servers = as.integer(n_servers),
    capacity = as.integer(capacity),
    service_rate = service_rate,
    busy = 0L,
    waiting = integer(),
    queue = data.frame(time = numeric(), type = character(), entity = integer()),
    counts = c(arrival = 0L, service_start = 0L, departure = 0L, balk = 0L),
    arrived_at = numeric(),
    waited = numeric(),
    sojourn = numeric(),
    balked = integer(),
    area_queue = 0,
    area_busy = 0,
    log = list()
  )
  class(state) <- "sim_state"
  return(state)
}

#' Check the internal consistency of a simulation state
#'
#' @param state A `sim_state` list as returned by [new_sim_state()].
#' @return `state`, invisibly. Signals an error naming the first malformed
#'   field among `time`, `busy`, `waiting`, `queue` and `counts`.
#' @keywords internal
validate_state <- function(state) {
  fields <- c("time", "busy", "waiting", "queue", "counts")
  absent <- setdiff(fields, names(state))
  if (length(absent) > 0L) {
    stop(sprintf("State is missing field `%s`.", absent[[1L]]), call. = FALSE)
  }

  # Check `time`
  if (!is.numeric(state$time) || length(state$time) != 1L || is.na(state$time)) {
    stop(sprintf("State field `%s` is malformed.", "time"), call. = FALSE)
  }
  # Check `busy`
  if (state$busy < 0L || state$busy > state$n_servers) {
    stop(sprintf("State field `%s` is malformed.", "busy"), call. = FALSE)
  }
  # Check `waiting`
  if (anyDuplicated(state$waiting) > 0L || length(state$waiting) > state$capacity) {
    stop(sprintf("State field `%s` is malformed.", "waiting"), call. = FALSE)
  }
  # Check `queue`
  if (!is.data.frame(state$queue) || is.unsorted(state$queue$time)) {
    stop(sprintf("State field `%s` is malformed.", "queue"), call. = FALSE)
  }
  # Check `counts`
  if (anyNA(state$counts) || any(state$counts < 0L)) {
    stop(sprintf("State field `%s` is malformed.", "counts"), call. = FALSE)
  }
  return(invisible(state))
}

#' Remove the next event from the queue
#'
#' @inheritParams validate_state
#' @return A list with elements `state` (without the event) and `event`
#'   (a list with `time`, `type` and `entity`, or `NULL` when the queue is
#'   empty).
#' @keywords internal
pop_event <- function(state) {
  if (nrow(state$queue) == 0L) {
    return(list(state = state, event = NULL))
  }
  event <- as.list(state$queue[1L, , drop = FALSE])
  state$queue <- state$queue[-1L, , drop = FALSE]
  return(list(state = state, event = event))
}

#' Take a snapshot of the simulation state
#'
#' @inheritParams validate_state
#' @param fields Fields to include, any of `"time"`, `"busy"`, `"waiting"`,
#'   `"scheduled"`, `"served"` and `"balked"`.
#' @return A named list with one element per requested field: the clock,
#'   the number of busy servers, the length of the waiting line, the number
#'   of scheduled events, the number of departures and of balked entities.
#' @export
state_snapshot <- function(state, fields = c("time", "busy", "waiting", "scheduled",
                                             "served", "balked")) {
  fields <- match.arg(fields, several.ok = TRUE)
  snapshot <- list()

  if ("time" %in% fields) {
    snapshot$time <- state$time
  }
  if ("busy" %in% fields) {
    snapshot$busy <- state$busy
  }
  if ("waiting" %in% fields) {
    snapshot$waiting <- length(state$waiting)
  }
  if ("scheduled" %in% fields) {
    snapshot$scheduled <- nrow(state$queue)
  }
  if ("served" %in% fields) {
    snapshot$served <- state$counts[["departure"]]
  }
  if ("balked" %in% fields) {
    snapshot$balked <- length(state$balked)
  }
  return(snapshot)
}

#' Drop scheduled events
#'
#' Removes pending events of one type, for example the departure of an
#' entity that leaves the system early. Only events between `after` and
#' `before` are considered.
#'
#' @inheritParams validate_state
#' @param type Event type to drop.
#' @param entity Optional integer ids; when `NULL`, events of all entities
#'   are dropped.
#' @param after,before Optional time bounds (inclusive).
#' @return The updated `state`.
#' @export
drop_events <- function(state, type, entity = NULL, after = NULL, before = NULL) {
  hit <- state$queue$type == type
  if (!is.null(entity)) {
    hit <- hit & state$queue$entity %in% entity
  }

  # Lower time bound
  if (!is.null(after)) {
    if (!is.numeric(after) || length(after) != 1L) {
      stop("`after` must be a single number.", call. = FALSE)
    }
    hit <- hit & state$queue$time >= after
  }
  # Upper time bound
  if (!is.null(before)) {
    if (!is.numeric(before) || length(before) != 1L) {
      stop("`before` must be a single number.", call. = FALSE)
    }
    hit <- hit & state$queue$time <= before
  }
  state$queue <- state$queue[!hit, , drop = FALSE]
  rownames(state$queue) <- NULL
  return(state)
}

#' Check Little's law on a finished run
#'
#' @param state A finished `sim_state`, see `run_queue_sim()`.
#' @return A list with elements `L` (mean number in system), `lambda`
#'   (accepted arrival rate), `W` (mean sojourn of served entities) and
#'   `gap` (relative difference between `L` and `lambda * W`).
#' @export
little_check <- function(state) {
  horizon <- state$time - state$start_time
  served <- setdiff(names(state$sojourn), as.character(state$balked))
  L <- (state$area_queue + state$area_busy) / horizon
  lambda <- length(served) / horizon
  W <- mean(state$sojourn[served])
  gap <- abs(L - lambda * W) / max(L, .Machine$double.eps)
  return(list(L = L, lambda = lambda, W = W, gap = gap))
}

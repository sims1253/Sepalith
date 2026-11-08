#' Create an event
#'
#' @param time Time at which the event fires.
#' @param type Event type, one of the names of [event_handlers()].
#' @param entity Integer id of the entity the event concerns.
#' @return A one-row data frame with columns `time`, `type` and `entity`.
#' @keywords internal
new_event <- function(time, type, entity) {
  # Validate `time`
  if (!is.numeric(time) || length(time) != 1L || is.na(time)) {
    stop(sprintf("Event field `%s` must be a single non-missing value.", "time"),
      call. = FALSE)
  }
  # Validate `type`
  if (!is.character(type) || length(type) != 1L || is.na(type)) {
    stop(sprintf("Event field `%s` must be a single non-missing value.", "type"),
      call. = FALSE)
  }
  # Validate `entity`
  if (!is.numeric(entity) || length(entity) != 1L || is.na(entity)) {
    stop(sprintf("Event field `%s` must be a single non-missing value.", "entity"),
      call. = FALSE)
  }

  event <- data.frame(time = time, type = type, entity = as.integer(entity))
  return(event)
}

#' Schedule an event
#'
#' @param state A `sim_state` list as returned by [new_sim_state()].
#' @inheritParams new_event
#' @return The updated `state` with the event queue kept sorted by time.
#' @keywords internal
schedule_event <- function(state, time, type, entity) {
  queue <- rbind(state$queue, new_event(time, type, entity))
  state$queue <- queue[order(queue$time), , drop = FALSE]
  rownames(state$queue) <- NULL
  return(state)
}

#' Event handlers for the single-queue model
#'
#' Each handler accumulates the time-weighted queue length and number of
#' busy servers, advances the clock to the event time, updates the counter
#' for its event type, changes the state and appends a row to the log.
#'
#' @param state A `sim_state` list as returned by [new_sim_state()].
#' @param event An event list with `time`, `type` and `entity`.
#' @return The updated `state`.
#' @rdname event-handlers
#' @family event handlers
#' @keywords internal
handle_arrival <- function(state, event) {
  elapsed <- event$time - state$time
  state$area_queue <- state$area_queue + elapsed * length(state$waiting)
  state$area_busy <- state$area_busy + elapsed * state$busy
  state$time <- event$time
  state$counts[["arrival"]] <- state$counts[["arrival"]] + 1L
  key <- as.character(event$entity)

  # A full waiting line turns the arrival away
  if (length(state$waiting) >= state$capacity) {
    state <- schedule_event(state, event$time, "balk", event$entity)
  } else {
    state$waiting <- c(state$waiting, event$entity)
    state$arrived_at[[key]] <- event$time
    if (state$busy < state$n_servers) {
      state <- schedule_event(state, event$time, "service_start", event$entity)
    }
  }

  entry <- data.frame(time = event$time, type = "arrival", entity = event$entity,
    busy = state$busy, waiting = length(state$waiting))
  state$log[[length(state$log) + 1L]] <- entry
  return(state)
}

#' @rdname event-handlers
#' @keywords internal
handle_service_start <- function(state, event) {
  elapsed <- event$time - state$time
  state$area_queue <- state$area_queue + elapsed * length(state$waiting)
  state$area_busy <- state$area_busy + elapsed * state$busy
  state$time <- event$time
  if (state$busy >= state$n_servers || length(state$waiting) == 0L) {
    return(state)
  }
  state$counts[["service_start"]] <- state$counts[["service_start"]] + 1L
  event$entity <- state$waiting[[1L]]
  key <- as.character(event$entity)

  # The head of the line takes the free server
  state$waiting <- state$waiting[-1L]
  state$busy <- state$busy + 1L
  state$waited[[key]] <- event$time - state$arrived_at[[key]]
  duration <- stats::rexp(1L, rate = state$service_rate)
  state <- schedule_event(state, event$time + duration, "departure", event$entity)

  entry <- data.frame(time = event$time, type = "service_start", entity = event$entity,
    busy = state$busy, waiting = length(state$waiting))
  state$log[[length(state$log) + 1L]] <- entry
  return(state)
}

#' @rdname event-handlers
#' @keywords internal
handle_departure <- function(state, event) {
  elapsed <- event$time - state$time
  state$area_queue <- state$area_queue + elapsed * length(state$waiting)
  state$area_busy <- state$area_busy + elapsed * state$busy
  state$time <- event$time
  state$counts[["departure"]] <- state$counts[["departure"]] + 1L
  key <- as.character(event$entity)

  # A freed server pulls the next entity from the line
  state$busy <- state$busy - 1L
  state$sojourn[[key]] <- event$time - state$arrived_at[[key]]
  if (length(state$waiting) > 0L) {
    state <- schedule_event(state, event$time, "service_start", state$waiting[[1L]])
  }

  entry <- data.frame(time = event$time, type = "departure", entity = event$entity,
    busy = state$busy, waiting = length(state$waiting))
  state$log[[length(state$log) + 1L]] <- entry
  return(state)
}

#' @rdname event-handlers
#' @keywords internal
handle_balk <- function(state, event) {
  elapsed <- event$time - state$time
  state$area_queue <- state$area_queue + elapsed * length(state$waiting)
  state$area_busy <- state$area_busy + elapsed * state$busy
  state$time <- event$time
  state$counts[["balk"]] <- state$counts[["balk"]] + 1L
  key <- as.character(event$entity)

  # A turned-away entity leaves without being served
  state$balked <- c(state$balked, event$entity)
  state$sojourn[[key]] <- 0

  entry <- data.frame(time = event$time, type = "balk", entity = event$entity,
    busy = state$busy, waiting = length(state$waiting))
  state$log[[length(state$log) + 1L]] <- entry
  return(state)
}

#' Dispatch table of event handlers
#'
#' @return A named list mapping the event types `arrival`, `service_start`,
#'   `departure` and `balk` to their handler functions.
#' @export
event_handlers <- function() {
  handlers <- list(
    arrival = handle_arrival,
    service_start = handle_service_start,
    departure = handle_departure,
    balk = handle_balk
  )
  return(handlers)
}

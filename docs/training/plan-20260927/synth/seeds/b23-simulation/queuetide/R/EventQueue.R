#' Create a simulation event
#'
#' An event is a small list holding the time it fires, a type label used for
#' dispatch and an optional payload such as a customer id.
#'
#' @param time Simulation time at which the event fires.
#' @param type Character label of the event, e.g. `"arrival"`.
#' @param payload Optional list of extra data carried by the event.
#' @return A list of class `qt_event`.
#' @export
newEvent = function(time, type, payload = list()) {
  stopifnot(is.numeric(time), length(time) == 1, !is.na(time))
  stopifnot(is.character(type), length(type) == 1)
  structure(
    list(time = time, type = type, payload = payload),
    class = "qt_event"
  )
}

#' Test whether an object is an event
#'
#' @param x Object to test.
#' @return `TRUE` if `x` inherits from `qt_event`, otherwise `FALSE`.
#' @export
isEvent = function(x) {
  inherits(x, "qt_event")
}

#' Format an event for display
#'
#' @param x A `qt_event` object.
#' @param digits Number of significant digits used for the event time.
#' @param ... Ignored.
#' @return A single character string.
#' @export
format.qt_event = function(x, digits = 4, ...) {
  extra = if (length(x$payload) > 0) {
    paste0(" {", paste(names(x$payload), collapse = ", "), "}")
  } else {
    ""
  }
  paste0("[t=", format(x$time, digits = digits), "] ", x$type, extra)
}

#' Print an event
#'
#' @param x A `qt_event` object.
#' @param ... Passed to [format.qt_event()].
#' @return `x`, invisibly.
#' @export
print.qt_event = function(x, ...) {
  cat(format(x, ...), "\n", sep = "")
  invisible(x)
}

#' Find the insertion position for a new event time
#'
#' Binary search over a sorted vector of event times. Ties are placed after
#' existing events with the same time so that the queue is first in, first
#' out for simultaneous events.
#'
#' @param times Sorted numeric vector of queued event times.
#' @param time Time of the event to insert.
#' @return The index at which the new event should be inserted.
#' @keywords internal
findInsertPos = function(times, time) {
  lo = 1L
  hi = length(times) + 1L
  while (lo < hi) {
    mid = (lo + hi) %/% 2L
    if (times[mid] <= time) {
      lo = mid + 1L
    } else {
      hi = mid
    }
  }
  lo
}

#' Future event list
#'
#' A priority queue of [newEvent()] objects ordered by time. Used by the
#' simulation loop to pull the next event to process.
#'
#' @export
EventQueue = R6::R6Class("EventQueue",
  public = list(
    #' @description Create an empty event queue.
    #' @return A new `EventQueue` object.
    initialize = function() {
      private$events = list()
      private$times = numeric(0)
      invisible(self)
    },
    #' @description Insert an event, keeping the queue sorted by time.
    #' @param event A `qt_event` object.
    #' @return The queue, invisibly, for chaining.
    push = function(event) {
      if (!isEvent(event)) {
        stop("`event` must be created with newEvent().", call. = FALSE)
      }
      pos = findInsertPos(private$times, event$time)
      private$events = append(private$events, list(event), after = pos - 1L)
      private$times = append(private$times, event$time, after = pos - 1L)
      invisible(self)
    },
    #' @description Remove and return the earliest event.
    #' @return A `qt_event` object, or `NULL` if the queue is empty.
    pop = function() {
      if (self$isEmpty()) {
        return(NULL)
      }
      event = private$events[[1]]
      private$events = private$events[-1]
      private$times = private$times[-1]
      event
    },
    #' @description Return the earliest event without removing it.
    #' @return A `qt_event` object, or `NULL` if the queue is empty.
    peek = function() {
      if (self$isEmpty()) {
        return(NULL)
      }
      private$events[[1]]
    },
    #' @description Number of queued events.
    #' @return An integer.
    size = function() {
      length(private$events)
    },
    #' @description Whether the queue has no events.
    #' @return `TRUE` or `FALSE`.
    isEmpty = function() {
      length(private$events) == 0L
    },
    #' @description Drop all events of a given type, e.g. to cancel
    #'   pending departures when a server breaks down.
    #' @param type Character label of the events to drop.
    #' @return The number of events removed.
    cancel = function(type) {
      keep = vapply(private$events, function(e) e$type != type, logical(1))
      removed = sum(!keep)
      private$events = private$events[keep]
      private$times = private$times[keep]
      removed
    },
    #' @description Print the queue length and the next few events.
    #' @param n Number of upcoming events to show.
    #' @return The queue, invisibly.
    print = function(n = 5) {
      cat("<EventQueue> ", self$size(), " pending event(s)\n", sep = "")
      shown = utils::head(private$events, n)
      for (event in shown) {
        cat("  ", format(event), "\n", sep = "")
      }
      if (self$size() > n) {
        cat("  ... and ", self$size() - n, " more\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    events = NULL,
    times = NULL
  )
)

#' Build an event queue from a vector of times
#'
#' Convenience constructor that pushes one event of the same type for every
#' time in `times`, attaching a sequential id to each payload.
#'
#' @param times Numeric vector of event times, in any order.
#' @param type Character label of the events, e.g. `"arrival"`.
#' @return An [EventQueue] object.
#' @export
eventQueueFromTimes = function(times, type = "arrival") {
  queue = EventQueue$new()
  for (i in seq_along(times)) {
    queue$push(newEvent(times[[i]], type, payload = list(id = i)))
  }
  queue
}

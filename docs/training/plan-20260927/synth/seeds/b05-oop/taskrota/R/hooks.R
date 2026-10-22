#' Supported hook events
#'
#' @return A character vector of event names a [HookRegistry] accepts.
#' @keywords internal
hook_events = function() {
  c("job_started", "job_succeeded", "job_failed", "queue_empty")
}

#' Register and fire callbacks
#'
#' A `HookRegistry` stores named callbacks for scheduler events. Callbacks
#' receive the event name and a payload list.
#'
#' @export
HookRegistry = R6::R6Class(
  "HookRegistry",
  public = list(
    #' @description
    #' Create an empty registry.
    initialize = function() {
      private$.hooks = stats::setNames(
        rep(list(list()), length(hook_events())),
        hook_events()
      )
      invisible(self)
    },
    #' @description
    #' Register a callback for an event.
    #' @param event One of the supported event names.
    #' @param callback A function taking `event` and `payload` arguments.
    #' @param id A name for the callback, used by `off()`.
    #' @return The registry, invisibly.
    on = function(event, callback, id = NULL) {
      event = match.arg(event, hook_events())
      stopifnot(is.function(callback))
      id = id %||% paste0("hook", length(private$.hooks[[event]]) + 1L)
      private$.hooks[[event]][[id]] = callback
      invisible(self)
    },
    #' @description
    #' Remove a callback.
    #' @param event One of the supported event names.
    #' @param id The name given to the callback when it was registered.
    #' @return The registry, invisibly.
    off = function(event, id) {
      event = match.arg(event, hook_events())
      private$.hooks[[event]][[id]] = NULL
      invisible(self)
    },
    #' @description
    #' Call every callback registered for an event.
    #' @param event One of the supported event names.
    #' @param payload A list passed to each callback.
    #' @return A list of the callback return values, invisibly.
    emit = function(event, payload = list()) {
      event = match.arg(event, hook_events())
      callbacks = private$.hooks[[event]]
      results = vector("list", length(callbacks))
      names(results) = names(callbacks)
      for (id in names(callbacks)) {
        results[[id]] = tryCatch(
          callbacks[[id]](event, payload),
          error = function(e) {
            warning("Hook '", id, "' failed on ", event, ": ", conditionMessage(e),
                    call. = FALSE)
            NULL
          }
        )
      }
      invisible(results)
    },
    #' @description
    #' Count the callbacks registered for each event.
    #' @return A named integer vector.
    count = function() {
      vapply(private$.hooks, length, integer(1))
    },
    #' @description
    #' Print the registry.
    #' @param ... Ignored.
    print = function(...) {
      cat("<HookRegistry>\n")
      counts = self$count()
      for (event in names(counts)) {
        cat("  ", format(event, width = 14), counts[[event]], "\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    .hooks = NULL
  )
)

#' Hook that writes events to a log file
#'
#' @param path Path of the log file. Lines are appended.
#' @param fields Payload fields to include on each line.
#' @return A callback suitable for `HookRegistry$on()`.
#' @family hook factories
#' @export
hook_log_to_file = function(path, fields = c("name", "status")) {
  force(path)
  force(fields)
  function(event, payload) {
    values = vapply(fields, function(f) as.character(payload[[f]] %||% NA), character(1))
    line = paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), event,
                 paste(fields, values, sep = "=", collapse = " "))
    cat(line, "\n", file = path, append = TRUE, sep = "")
    invisible(line)
  }
}

#' Hook that reports events on the console
#'
#' @param fields Payload fields to include in the message.
#' @param quiet_events Events that should not produce a message.
#' @return A callback suitable for `HookRegistry$on()`.
#' @family hook factories
#' @export
hook_console = function(fields = c("name", "status"), quiet_events = character()) {
  force(fields)
  force(quiet_events)
  function(event, payload) {
    if (event %in% quiet_events) {
      return(invisible(NULL))
    }
    values = vapply(fields, function(f) as.character(payload[[f]] %||% NA), character(1))
    msg = paste0("[", event, "] ", paste(fields, values, sep = ": ", collapse = ", "))
    message(msg)
    invisible(msg)
  }
}

#' Hook that collects job timings
#'
#' The returned callback keeps a running record of elapsed times. Call it with
#' `event = "report"` to retrieve the collected values.
#'
#' @param fields Payload fields to record alongside the elapsed time.
#' @return A callback suitable for `HookRegistry$on()`.
#' @family hook factories
#' @export
hook_collect_timings = function(fields = "name") {
  force(fields)
  store = list()
  function(event, payload) {
    if (identical(event, "report")) {
      return(do.call(rbind, lapply(store, as.data.frame, stringsAsFactors = FALSE)))
    }
    record = lapply(stats::setNames(fields, fields), function(f) payload[[f]] %||% NA)
    record$event = event
    record$elapsed = payload$elapsed %||% NA_real_
    store[[length(store) + 1L]] <<- record
    invisible(length(store))
  }
}

#' Attach a hook registry to a scheduler run
#'
#' Runs every job queued on a scheduler and emits the matching events on the
#' registry as jobs start and finish.
#'
#' @param scheduler A [Scheduler].
#' @param hooks A [HookRegistry].
#' @return The scheduler, invisibly.
#' @export
run_with_hooks = function(scheduler, hooks) {
  stopifnot(inherits(scheduler, "Scheduler"), inherits(hooks, "HookRegistry"))
  while (!scheduler$queue$is_empty) {
    upcoming = scheduler$queue$peek()
    hooks$emit("job_started", list(name = upcoming$name, id = upcoming$id))
    job = scheduler$run_next()
    payload = list(name = job$name, id = job$id, status = job$status, elapsed = job$elapsed)
    if (job$status == "succeeded") {
      hooks$emit("job_succeeded", payload)
    } else {
      hooks$emit("job_failed", payload)
    }
  }
  hooks$emit("queue_empty", list(name = NA, status = NA))
  invisible(scheduler)
}

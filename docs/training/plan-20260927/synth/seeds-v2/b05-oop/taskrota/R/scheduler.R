#' Retry backoff policies
#'
#' @description Each factory returns a function of the attempt number giving
#'   the delay in seconds before the next retry. `backoff_constant()` waits
#'   the same time before every retry, `backoff_linear()` adds `base` seconds
#'   per attempt and `backoff_exponential()` doubles the delay, with optional
#'   jitter to avoid synchronised retries.
#' @param base Delay in seconds.
#' @param max_delay Upper bound on any single delay, in seconds.
#' @return A function of the attempt number returning a delay in seconds.
#' @export
backoff_constant <- function(base = 1, max_delay = 60) {
  force(base)
  force(max_delay)
  function(attempt) {
    min(base, max_delay)
  }
}

#' @rdname backoff_constant
#' @export
backoff_linear <- function(base = 1, max_delay = 60) {
  force(base)
  force(max_delay)
  function(attempt) {
    min(base * attempt, max_delay)
  }
}

#' @rdname backoff_constant
#' @param jitter Fraction of the delay to randomise, between 0 and 1.
#' @export
backoff_exponential <- function(base = 1, max_delay = 60, jitter = 0) {
  stopifnot(jitter >= 0, jitter <= 1)
  force(base)
  force(max_delay)
  function(attempt) {
    delay <- min(base * 2^(attempt - 1), max_delay)
    delay * (1 - jitter * stats::runif(1))
  }
}

#' Format a duration for display
#'
#' @param seconds A numeric vector of durations in seconds.
#' @return A character vector such as `"850ms"`, `"12.4s"` or `"3m 05s"`.
#' @keywords internal
format_duration <- function(seconds) {
  out <- character(length(seconds))
  for (i in seq_along(seconds)) {
    s <- seconds[[i]]
    out[[i]] <- if (is.na(s)) {
      "--"
    } else if (s < 1) {
      paste0(round(s * 1000), "ms")
    } else if (s < 60) {
      paste0(format(round(s, 1), nsmall = 1), "s")
    } else {
      # Round first, otherwise 119.7s would come out as "1m 60s"
      s <- round(s)
      sprintf("%dm %02ds", as.integer(s %/% 60), as.integer(s %% 60))
    }
  }
  out
}

#' Run queued jobs with retries
#'
#' A `Scheduler` pulls jobs from a [PriorityQueue], runs them, and re-queues
#' failed jobs according to a backoff policy until they succeed or run out of
#' attempts.
#'
#' @export
Scheduler <- R6::R6Class(
  "Scheduler",
  public = list(
    #' @field queue The [PriorityQueue] the scheduler serves.
    queue = NULL,
    #' @field max_attempts Maximum number of runs per job.
    max_attempts = 3L,
    #' @description
    #' Create a new scheduler.
    #' @param queue A [PriorityQueue]. A new empty queue is used by default.
    #' @param max_attempts Maximum number of runs per job.
    #' @param backoff A backoff policy such as [backoff_exponential()].
    #' @param sleep Function used to wait between retries.
    initialize = function(queue = PriorityQueue$new(), max_attempts = 3L,
                          backoff = backoff_exponential(), sleep = Sys.sleep) {
      stopifnot(inherits(queue, "PriorityQueue"), is.function(backoff), is.function(sleep))
      self$queue <- queue
      self$max_attempts <- as.integer(max_attempts)

      private$.backoff <- backoff
      private$.sleep <- sleep
      private$.attempts <- list()
      invisible(self)
    },
    #' @description
    #' Add a job to the queue.
    #' @param job A [Job] or anything accepted by [as_job()].
    #' @param ... Passed to [as_job()].
    #' @return The scheduler, invisibly.
    submit = function(job, ...) {
      self$queue$push(job, ...)
      invisible(self)
    },
    #' @description
    #' Run the next job in the queue.
    #' @return The job that was run, or `NULL` if the queue was empty.
    run_next = function() {
      job <- self$queue$pop()
      if (is.null(job)) {
        return(NULL)
      }

      attempt <- (private$.attempts[[job$id]] %||% 0L) + 1L
      private$.attempts[[job$id]] <- attempt
      job$run()
      private$log_run(job, attempt)

      # A retried job goes back in pending; the history keeps the failure
      if (job$status == "failed" && attempt < self$max_attempts) {
        private$.sleep(private$.backoff(attempt))
        job$reset()
        self$queue$push(job)
      }
      job
    },
    #' @description
    #' Run jobs until the queue is empty.
    #' @param limit Maximum number of runs, as a safety net.
    #' @return The scheduler, invisibly.
    run_all = function(limit = 1000L) {
      runs <- 0L
      while (!self$queue$is_empty && runs < limit) {
        self$run_next()
        runs <- runs + 1L
      }

      if (runs >= limit && !self$queue$is_empty) {
        warning("Stopped after ", limit, " runs with jobs still queued.", call. = FALSE)
      }
      invisible(self)
    },
    #' @description
    #' Print the scheduler state.
    #' @param ... Ignored.
    print = function(...) {
      hist <- self$history
      cat("<Scheduler> ", self$queue$size, " queued, ", nrow(hist), " run(s)\n", sep = "")
      if (nrow(hist) > 0L) {
        tally <- table(factor(hist$status, levels = c("succeeded", "failed")))
        cat("  succeeded: ", tally[["succeeded"]], ", failed: ", tally[["failed"]], "\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    .backoff = NULL,
    .sleep = NULL,
    .attempts = NULL,
    .log = NULL,
    log_run = function(job, attempt) {
      entry <- data.frame(
        id = job$id,
        name = job$name,
        attempt = attempt,
        status = job$status,
        elapsed = job$elapsed,
        stringsAsFactors = FALSE
      )
      private$.log <- rbind(private$.log, entry)
      invisible(NULL)
    }
  ),
  active = list(
    #' @field history A data frame with one row per run.
    history = function(value) {
      if (!missing(value)) {
        stop("`history` is read-only.", call. = FALSE)
      }
      if (is.null(private$.log)) {
        return(data.frame(
          id = character(), name = character(), attempt = integer(),
          status = character(), elapsed = numeric(), stringsAsFactors = FALSE
        ))
      }
      private$.log
    }
  )
)

#' Summarise a scheduler's run history
#'
#' @description Collapses the run log to one row per job, keeping the final
#'   status, the number of attempts and the total run time.
#' @param scheduler A [Scheduler].
#' @return A data frame with columns `name`, `attempts`, `status` and `time`.
#' @export
scheduler_report <- function(scheduler) {
  stopifnot(inherits(scheduler, "Scheduler"))
  hist <- scheduler$history
  if (nrow(hist) == 0L) {
    return(data.frame(name = character(), attempts = integer(), status = character(),
                      time = character(), stringsAsFactors = FALSE))
  }

  # Keep jobs in the order they first ran, not alphabetically by id
  by_job <- split(hist, factor(hist$id, levels = unique(hist$id)))
  rows <- lapply(by_job, function(runs) {
    data.frame(
      name = runs$name[[1L]],
      attempts = max(runs$attempt),
      status = runs$status[[nrow(runs)]],
      time = format_duration(sum(runs$elapsed, na.rm = TRUE)),
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

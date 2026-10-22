#' Allowed job status values
#'
#' Returns the ordered set of states a job can move through. The order is
#' used when sorting job listings for display.
#'
#' @return A character vector of status labels.
#' @keywords internal
job_status_levels <- function() {
  c("pending", "running", "succeeded", "failed", "cancelled")
}

#' Generate a job identifier
#'
#' Builds a short, mostly unique identifier from the current time and a
#' random suffix.
#'
#' @param prefix A string placed in front of the identifier.
#' @return A single string.
#' @keywords internal
new_job_id <- function(prefix = "job") {
  stamp <- format(Sys.time(), "%Y%m%d%H%M%S")
  suffix <- paste(sample(c(letters, 0:9), 6, replace = TRUE), collapse = "")
  paste(prefix, stamp, suffix, sep = "-")
}

#' Check a job priority
#'
#' Validates that a priority is a single whole number between 1 and 10 and
#' returns it as an integer.
#'
#' @param priority The priority to check.
#' @param arg The argument name used in error messages.
#' @return `priority`, coerced to integer.
#' @keywords internal
validate_priority <- function(priority, arg = "priority") {
  if (!is.numeric(priority) || length(priority) != 1L || is.na(priority)) {
    stop("`", arg, "` must be a single number.", call. = FALSE)
  }
  if (priority != round(priority)) {
    stop("`", arg, "` must be a whole number.", call. = FALSE)
  }
  if (priority < 1 || priority > 10) {
    stop("`", arg, "` must be between 1 and 10, not ", priority, ".", call. = FALSE)
  }
  as.integer(priority)
}

#' A unit of work
#'
#' A `Job` wraps a function and its arguments together with a priority and
#' bookkeeping about when it ran and how it finished.
#'
#' @export
Job <- R6::R6Class(
  "Job",
  public = list(
    #' @field id Unique identifier of the job.
    id = NULL,
    #' @field name Human readable label.
    name = NULL,
    #' @field priority Integer priority from 1 (low) to 10 (high).
    priority = NULL,
    #' @field status Current status, one of [job_status_levels()].
    status = "pending",
    #' @description
    #' Create a new job.
    #' @param fn The function to run.
    #' @param args A list of arguments passed to `fn`.
    #' @param name A label for the job. Defaults to the identifier.
    #' @param priority Integer priority from 1 to 10.
    initialize = function(fn, args = list(), name = NULL, priority = 5L) {
      stopifnot(is.function(fn), is.list(args))
      self$id <- new_job_id()
      self$name <- name %||% self$id
      self$priority <- validate_priority(priority)
      private$.fn <- fn
      private$.args <- args
      invisible(self)
    },
    #' @description
    #' Run the job, capturing the result or the error.
    #' @return The job, invisibly.
    run = function() {
      if (self$status != "pending") {
        stop("Job '", self$name, "' has already been run.", call. = FALSE)
      }
      self$status <- "running"
      private$.started <- Sys.time()
      outcome <- tryCatch(
        list(value = do.call(private$.fn, private$.args), error = NULL),
        error = function(e) list(value = NULL, error = e)
      )
      private$.finished <- Sys.time()
      private$.result <- outcome$value
      private$.error <- outcome$error
      self$status <- if (is.null(outcome$error)) "succeeded" else "failed"
      invisible(self)
    },
    #' @description
    #' Put the job back into the pending state so it can run again.
    #' @return The job, invisibly.
    reset = function() {
      self$status <- "pending"
      private$.result <- NULL
      private$.error <- NULL
      private$.started <- NULL
      private$.finished <- NULL
      invisible(self)
    },
    #' @description
    #' Cancel a pending job.
    #' @return The job, invisibly.
    cancel = function() {
      if (self$status == "pending") {
        self$status <- "cancelled"
      }
      invisible(self)
    },
    #' @description
    #' Change the priority of the job.
    #' @param priority Integer priority from 1 to 10.
    #' @return The job, invisibly.
    set_priority = function(priority) {
      self$priority <- validate_priority(priority)
      invisible(self)
    },
    #' @description
    #' Retrieve the value returned by the job function.
    #' @return The stored result, or an error if the job failed.
    result = function() {
      if (self$status == "failed") {
        stop("Job '", self$name, "' failed: ", conditionMessage(private$.error), call. = FALSE)
      }
      private$.result
    },
    #' @description
    #' Print a one-line summary of the job.
    #' @param ... Ignored.
    print = function(...) {
      cat("<Job> ", self$name, " [", self$status, ", priority ", self$priority, "]\n", sep = "")
      if (!is.na(self$elapsed)) {
        cat("  elapsed: ", format(round(self$elapsed, 3)), "s\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    .fn = NULL,
    .args = NULL,
    .result = NULL,
    .error = NULL,
    .started = NULL,
    .finished = NULL
  ),
  active = list(
    #' @field elapsed Run time in seconds, or `NA` if the job has not finished.
    elapsed = function(value) {
      if (!missing(value)) {
        stop("`elapsed` is read-only.", call. = FALSE)
      }
      if (is.null(private$.started) || is.null(private$.finished)) {
        return(NA_real_)
      }
      as.numeric(difftime(private$.finished, private$.started, units = "secs"))
    },
    #' @field is_done `TRUE` once the job has left the pending and running states.
    is_done = function(value) {
      if (!missing(value)) {
        stop("`is_done` is read-only.", call. = FALSE)
      }
      self$status %in% c("succeeded", "failed", "cancelled")
    }
  )
)

#' Test whether an object is a job
#'
#' @param x An object to test.
#' @return `TRUE` if `x` inherits from `Job`, otherwise `FALSE`.
#' @export
is_job <- function(x) {
  inherits(x, "Job")
}

#' Coerce an object to a job
#'
#' Turns functions and formulas into [Job] objects; jobs are returned as is.
#'
#' @param x An object to coerce.
#' @param ... Passed on to the `Job$new()` constructor.
#' @return A `Job` object.
#' @export
as_job <- function(x, ...) {
  UseMethod("as_job")
}

#' @rdname as_job
#' @export
as_job.Job <- function(x, ...) {
  x
}

#' @rdname as_job
#' @export
as_job.function <- function(x, ...) {
  Job$new(fn = x, ...)
}

#' @rdname as_job
#' @export
as_job.formula <- function(x, ...) {
  if (length(x) != 2L) {
    stop("A job formula must be one-sided, like `~ expr`.", call. = FALSE)
  }
  env <- environment(x)
  body <- x[[2L]]
  fn <- function() eval(body, envir = env)
  Job$new(fn = fn, ...)
}

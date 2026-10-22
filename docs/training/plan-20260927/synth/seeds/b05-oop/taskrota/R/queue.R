#' A priority queue of jobs
#'
#' `PriorityQueue` keeps jobs ordered by priority, highest first. Jobs with
#' equal priority are served in the order they were pushed.
#'
#' @export
PriorityQueue <- R6::R6Class(
  "PriorityQueue",
  public = list(
    #' @description
    #' Create an empty queue.
    #' @param capacity Maximum number of jobs the queue may hold.
    initialize = function(capacity = Inf) {
      stopifnot(is.numeric(capacity), length(capacity) == 1L, capacity > 0)
      private$.capacity <- capacity
      private$.jobs <- list()
      private$.seq <- integer()
      invisible(self)
    },
    #' @description
    #' Add a job to the queue.
    #' @param job A [Job] or anything accepted by [as_job()].
    #' @param ... Passed to [as_job()].
    #' @return The queue, invisibly, so calls can be chained.
    push = function(job, ...) {
      if (self$size >= private$.capacity) {
        stop("Queue is full (capacity ", private$.capacity, ").", call. = FALSE)
      }
      job <- as_job(job, ...)
      private$.counter <- private$.counter + 1L
      private$.jobs[[length(private$.jobs) + 1L]] <- job
      private$.seq <- c(private$.seq, private$.counter)
      private$reorder()
      invisible(self)
    },
    #' @description
    #' Remove and return the job with the highest priority.
    #' @return A [Job], or `NULL` when the queue is empty.
    pop = function() {
      if (self$is_empty) {
        return(NULL)
      }
      job <- private$.jobs[[1L]]
      private$.jobs <- private$.jobs[-1L]
      private$.seq <- private$.seq[-1L]
      job
    },
    #' @description
    #' Return the job with the highest priority without removing it.
    #' @return A [Job], or `NULL` when the queue is empty.
    peek = function() {
      if (self$is_empty) {
        return(NULL)
      }
      private$.jobs[[1L]]
    },
    #' @description
    #' Remove a job by identifier.
    #' @param id The identifier of the job to remove.
    #' @return The queue, invisibly.
    remove = function(id) {
      ids <- vapply(private$.jobs, function(j) j$id, character(1))
      keep <- ids != id
      if (all(keep)) {
        warning("No job with id '", id, "' in the queue.", call. = FALSE)
      }
      private$.jobs <- private$.jobs[keep]
      private$.seq <- private$.seq[keep]
      invisible(self)
    },
    #' @description
    #' Drop every job from the queue.
    #' @return The queue, invisibly.
    clear = function() {
      private$.jobs <- list()
      private$.seq <- integer()
      invisible(self)
    },
    #' @description
    #' Print the queue and its first few jobs.
    #' @param n Number of jobs to list.
    #' @param ... Ignored.
    print = function(n = 5L, ...) {
      cat("<PriorityQueue> ", self$size, " job(s)", sep = "")
      if (is.finite(private$.capacity)) {
        cat(" of ", private$.capacity, sep = "")
      }
      cat("\n")
      shown <- utils::head(private$.jobs, n)
      for (job in shown) {
        cat("  - ", job$name, " (priority ", job$priority, ")\n", sep = "")
      }
      if (self$size > n) {
        cat("  ... and ", self$size - n, " more\n", sep = "")
      }
      invisible(self)
    }
  ),
  private = list(
    .jobs = NULL,
    .seq = NULL,
    .capacity = Inf,
    .counter = 0L,
    reorder = function() {
      prios <- vapply(private$.jobs, function(j) j$priority, integer(1))
      ord <- order(-prios, private$.seq)
      private$.jobs <- private$.jobs[ord]
      private$.seq <- private$.seq[ord]
      invisible(NULL)
    }
  ),
  active = list(
    #' @field size Number of jobs currently queued.
    size = function(value) {
      if (!missing(value)) {
        stop("`size` is read-only.", call. = FALSE)
      }
      length(private$.jobs)
    },
    #' @field is_empty `TRUE` if the queue holds no jobs.
    is_empty = function(value) {
      if (!missing(value)) {
        stop("`is_empty` is read-only.", call. = FALSE)
      }
      length(private$.jobs) == 0L
    },
    #' @field jobs A copy of the queued jobs, in serving order.
    jobs = function(value) {
      if (!missing(value)) {
        stop("`jobs` is read-only; use `push()` and `pop()`.", call. = FALSE)
      }
      private$.jobs
    }
  )
)

#' Build a queue from a list of jobs
#'
#' Converts every element of a list with [as_job()] and pushes it onto a new
#' [PriorityQueue].
#'
#' @param jobs A list of jobs, functions or one-sided formulas.
#' @param capacity Maximum number of jobs the queue may hold.
#' @return A `PriorityQueue`.
#' @export
queue_from_list <- function(jobs, capacity = Inf) {
  if (!is.list(jobs)) {
    stop("`jobs` must be a list.", call. = FALSE)
  }
  queue <- PriorityQueue$new(capacity = capacity)
  nms <- names(jobs)
  for (i in seq_along(jobs)) {
    label <- if (is.null(nms) || !nzchar(nms[[i]])) NULL else nms[[i]]
    if (is_job(jobs[[i]])) {
      queue$push(jobs[[i]])
    } else {
      queue$push(jobs[[i]], name = label)
    }
  }
  queue
}

#' Run every job in a queue
#'
#' Pops jobs until the queue is empty, running each one in turn.
#'
#' @param queue A [PriorityQueue].
#' @param stop_on_error If `TRUE`, stop at the first failed job.
#' @return A list of the jobs that were run, in run order.
#' @export
queue_drain <- function(queue, stop_on_error = FALSE) {
  stopifnot(inherits(queue, "PriorityQueue"))
  done <- vector("list", queue$size)
  i <- 0L
  while (!queue$is_empty) {
    job <- queue$pop()
    job$run()
    i <- i + 1L
    done[[i]] <- job
    if (stop_on_error && job$status == "failed") {
      break
    }
  }
  done[seq_len(i)]
}

#' Tabulate the jobs in a queue
#'
#' @param queue A [PriorityQueue].
#' @return A data frame with one row per job and columns `id`, `name`,
#'   `priority` and `status`.
#' @export
queue_summary <- function(queue) {
  stopifnot(inherits(queue, "PriorityQueue"))
  jobs <- queue$jobs
  data.frame(
    id = vapply(jobs, function(j) j$id, character(1)),
    name = vapply(jobs, function(j) j$name, character(1)),
    priority = vapply(jobs, function(j) j$priority, integer(1)),
    status = vapply(jobs, function(j) j$status, character(1)),
    stringsAsFactors = FALSE
  )
}

#' Count queued jobs by priority
#'
#' @param queue A [PriorityQueue].
#' @return A named integer vector of counts, highest priority first.
#' @export
queue_priority_counts <- function(queue) {
  stopifnot(inherits(queue, "PriorityQueue"))
  prios <- vapply(queue$jobs, function(j) j$priority, integer(1))
  counts <- table(factor(prios, levels = 10:1))
  counts <- counts[counts > 0]
  stats::setNames(as.integer(counts), names(counts))
}

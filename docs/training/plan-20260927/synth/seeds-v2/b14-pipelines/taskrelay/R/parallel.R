#' Decide how many workers to use
#'
#' Starts from the number of available cores, keeps `reserve` cores free
#' for the rest of the system and caps the result at `max_workers`. The
#' `TASKRELAY_WORKERS` environment variable overrides all of this.
#'
#' @export
#' @param max_workers Upper limit on the number of workers.
#' @param reserve Number of cores to leave idle.
#' @return A single positive integer.
detect_workers <- function(max_workers = Inf, reserve = 1L) {
  env_value <- Sys.getenv("TASKRELAY_WORKERS", "")
  if (nzchar(env_value)) {
    n <- suppressWarnings(as.integer(env_value))
    if (!is.na(n)) {
      return(max(1L, n))
    }
    warning("Ignoring non-numeric TASKRELAY_WORKERS='", env_value, "'.", call. = FALSE)
  }

  # Physical cores: hyperthreads rarely help the I/O-heavy jobs we run
  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores)) {
    cores <- 1L
  }
  workers <- min(cores - reserve, max_workers)
  return(as.integer(max(1L, workers)))
}

#' Split indices into roughly equal chunks
#'
#' Used to hand each worker a contiguous block of work instead of one
#' element at a time, which keeps scheduling overhead low.
#'
#' @keywords internal
#' @param n Number of elements to split.
#' @param n_chunks Number of chunks wanted.
#' @return A list of integer vectors that together cover `seq_len(n)`.
chunk_indices <- function(n, n_chunks) {
  if (n == 0L) {
    return(list())
  }
  n_chunks <- min(n, max(1L, n_chunks))
  groups <- cut(seq_len(n), breaks = n_chunks, labels = FALSE)
  return(unname(split(seq_len(n), groups)))
}

#' Set a future plan for the current session
#'
#' Wraps [future::plan()] with a small set of named strategies and a worker
#' count taken from [detect_workers()] by default.
#'
#' @export
#' @param strategy One of `"sequential"`, `"multisession"` or
#'   `"multicore"`.
#' @param workers Number of workers for the parallel strategies.
#' @return The previous plan, invisibly, so it can be restored.
plan_workers <- function(strategy = c("sequential", "multisession", "multicore"),
                         workers = detect_workers()) {
  strategy <- match.arg(strategy)
  # Forking is unavailable on Windows and disabled inside RStudio
  if (strategy == "multicore" && !parallelly::supportsMulticore()) {
    warning("Multicore is not supported here; using multisession.", call. = FALSE)
    strategy <- "multisession"
  }

  old_plan <- switch(strategy,
    sequential = future::plan(future::sequential),
    multisession = future::plan(future::multisession, workers = workers),
    multicore = future::plan(future::multicore, workers = workers)
  )
  return(invisible(old_plan))
}

#' Evaluate code with a temporary future plan
#'
#' Sets the plan with [plan_workers()] and restores the previous plan when
#' `code` has finished, even if it fails.
#'
#' @export
#' @param code Code to evaluate.
#' @inheritParams plan_workers
#' @return The value of `code`.
with_workers <- function(code, strategy = "multisession", workers = detect_workers()) {
  old_plan <- plan_workers(strategy, workers = workers)
  on.exit(future::plan(old_plan), add = TRUE)
  # `code` is a promise; forcing it here runs it under the new plan
  return(force(code))
}

#' Map over a list with forked workers
#'
#' Uses [parallel::mclapply()] and turns `try-error` results into
#' conditions so callers can treat every backend the same way.
#'
#' @keywords internal
#' @param x A list or vector to iterate over.
#' @param fun A function applied to each element of `x`.
#' @param workers Number of workers to use.
#' @param ... Passed to `fun`.
#' @return A list the same length as `x`.
par_map_mclapply <- function(x, fun, workers, ...) {
  if (.Platform$OS.type == "windows") {
    workers <- 1L
  }

  out <- parallel::mclapply(x, fun, ..., mc.cores = workers, mc.preschedule = TRUE)
  failed <- vapply(out, inherits, logical(1), what = "try-error")
  out[failed] <- lapply(out[failed], function(e) attr(e, "condition"))

  names(out) <- names(x)
  return(out)
}

#' Map over a list with future workers
#'
#' Uses [furrr::future_map()] with a reproducible seed and wraps each call
#' so that errors come back as conditions instead of stopping the map.
#'
#' @keywords internal
#' @param x A list or vector to iterate over.
#' @param fun A function applied to each element of `x`.
#' @param workers Number of workers to use.
#' @param ... Passed to `fun`.
#' @return A list the same length as `x`.
par_map_furrr <- function(x, fun, workers, ...) {
  safe_fun <- function(el, ...) {
    tryCatch(fun(el, ...), error = function(e) e)
  }

  out <- with_workers(
    furrr::future_map(x, safe_fun, ..., .options = furrr::furrr_options(seed = TRUE)),
    strategy = "multisession",
    workers = workers
  )
  names(out) <- names(x)
  return(out)
}

#' Map over a list with a selectable backend
#'
#' A single entry point for the parallel helpers. Errors raised by `fun` are
#' returned as condition objects; use [collect_errors()] to inspect them.
#'
#' @export
#' @param x A list or vector to iterate over.
#' @param fun A function applied to each element of `x`.
#' @param backend One of `"sequential"`, `"mclapply"` or `"furrr"`.
#' @param workers Number of workers to use.
#' @param ... Passed to `fun`.
#' @return A list the same length as `x`.
par_map <- function(x, fun, backend = c("sequential", "mclapply", "furrr"),
                    workers = detect_workers(), ...) {
  backend <- match.arg(backend)
  if (length(x) == 0L) {
    return(list())
  }
  # Spinning up workers for a single element is pure overhead
  if (workers <= 1L || length(x) == 1L) {
    backend <- "sequential"
  }

  out <- switch(backend,
    sequential = lapply(x, function(el) tryCatch(fun(el, ...), error = function(e) e)),
    mclapply = par_map_mclapply(x, fun, workers = workers, ...),
    furrr = par_map_furrr(x, fun, workers = workers, ...)
  )
  return(out)
}

#' Collect errors from a parallel map
#'
#' Finds the elements of a [par_map()] result that are conditions and
#' returns their messages, keyed by position or name.
#'
#' @export
#' @param results A list returned by [par_map()].
#' @return A named character vector of error messages, possibly empty.
collect_errors <- function(results) {
  is_error <- vapply(results, inherits, logical(1), what = "error")
  if (!any(is_error)) {
    return(character())
  }

  # Unnamed inputs (and unnamed elements of partly named ones) fall back
  # to their position
  keys <- names(results)
  if (is.null(keys)) {
    keys <- as.character(seq_along(results))
  }
  keys[is.na(keys) | !nzchar(keys)] <- as.character(which(is.na(keys) | !nzchar(keys)))

  messages <- vapply(results[is_error], conditionMessage, character(1))
  return(stats::setNames(messages, keys[is_error]))
}

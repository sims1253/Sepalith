#' Is burrowlog running quietly?
#'
#' Messages are suppressed when the `burrowlog.quiet` option is `TRUE` or
#' when the `BURROWLOG_QUIET` environment variable is set to `"true"`.
#'
#' @return `TRUE` or `FALSE`.
#' @keywords internal
is_quiet <- function() {
  opt <- getOption("burrowlog.quiet")
  if (!is.null(opt)) {
    return(isTRUE(opt))
  }
  tolower(Sys.getenv("BURROWLOG_QUIET", "false")) %in% c("true", "1", "yes")
}

#' Emit burrowlog status messages
#'
#' Thin wrappers around cli that respect [is_quiet()] and tag every message
#' with a `burrowlog_message` class so callers can muffle them selectively.
#'
#' @param message A cli-formatted message, interpolated in the caller's frame.
#' @param .envir The environment used to interpolate `message`.
#' @return `NULL`, invisibly.
#' @family burrowlog messages
#' @keywords internal
burrow_inform <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(message, class = "burrowlog_message", .envir = .envir)
}

#' @rdname burrow_inform
burrow_success <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(c("v" = message), class = "burrowlog_message", .envir = .envir)
}

#' @rdname burrow_inform
burrow_warn <- function(message, .envir = parent.frame()) {
  if (is_quiet()) {
    return(invisible())
  }
  cli::cli_inform(c("!" = message), class = "burrowlog_message", .envir = .envir)
}

#' Format a named vector as cli bullets
#'
#' Turns a named vector such as `c(rows = 120, dropped = 3)` into aligned
#' bullet lines, suitable for passing to [cli::cli_bullets()].
#'
#' @param x A named atomic vector of summary values.
#' @param bullet The cli bullet type used for each line.
#' @return A named character vector of bullet lines.
#' @keywords internal
format_summary_bullets <- function(x, bullet = c("*", "i", "v", "x", "!")) {
  bullet <- match.arg(bullet)
  if (length(x) == 0L) {
    return(character())
  }
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    cli::cli_abort(
      "{.arg x} must be a fully named vector.",
      class = c("burrowlog_error_unnamed", "burrowlog_error")
    )
  }
  labels <- format(names(x), width = max(nchar(names(x))))
  values <- if (is.numeric(x)) format(x, big.mark = ",", trim = TRUE) else as.character(x)
  lines <- paste0(labels, " : ", values)
  # Escape braces so values are never interpolated by cli
  lines <- gsub("([{}])", "\\1\\1", lines)
  stats::setNames(lines, rep(bullet, length(lines)))
}

#' Apply a function to each element with a progress bar
#'
#' Like [lapply()], but shows a cli progress bar unless burrowlog is quiet.
#' Errors are re-signalled with the index of the failing element.
#'
#' @param x A vector or list to iterate over.
#' @param fn The function applied to each element.
#' @param label A short label shown next to the progress bar.
#' @param ... Passed to `fn`.
#' @return A list the same length as `x`.
#' @export
burrow_map <- function(x, fn, label = "Processing", ...) {
  fn <- rlang::as_function(fn)
  n <- length(x)
  out <- vector("list", n)
  names(out) <- names(x)
  if (!is_quiet()) {
    bar <- cli::cli_progress_bar(label, total = n, .envir = environment())
  }
  for (i in seq_len(n)) {
    out[[i]] <- tryCatch(
      fn(x[[i]], ...),
      error = function(cnd) {
        cli::cli_abort(
          "Failed while processing element {i} of {n}.",
          parent = cnd,
          class = c("burrowlog_error_map", "burrowlog_error"),
          index = i
        )
      }
    )
    if (!is_quiet()) {
      cli::cli_progress_update(id = bar)
    }
  }
  out
}

#' Run a block of code as a named step
#'
#' Prints a spinner-style step message while `code` runs, then reports the
#' elapsed time. The step is marked as failed if `code` errors.
#'
#' @param msg The step description.
#' @param code The code to evaluate.
#' @return The value of `code`.
#' @export
#' @examples
#' with_step("Counting rows", nrow(mtcars))
with_step <- function(msg, code) {
  if (is_quiet()) {
    return(code)
  }
  started <- Sys.time()
  cli::cli_progress_step(msg, msg_failed = paste(msg, "(failed)"))
  result <- code
  elapsed <- round(as.numeric(difftime(Sys.time(), started, units = "secs")), 1)
  cli::cli_progress_done()
  burrow_inform("{.emph {msg}} took {elapsed}s.")
  result
}

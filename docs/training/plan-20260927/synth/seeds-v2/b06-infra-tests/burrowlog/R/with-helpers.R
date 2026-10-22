#' Silence burrowlog messages temporarily
#'
#' `local_quiet()` sets the `burrowlog.quiet` option until the calling frame
#' exits; `with_quiet()` does the same for the duration of `code`.
#'
#' @param quiet Whether messages should be suppressed.
#' @param .local_envir The environment whose exit restores the previous state.
#' @return `local_quiet()` returns the previous option value, invisibly.
#' @export
local_quiet <- function(quiet = TRUE, .local_envir = parent.frame()) {
  stopifnot(is.logical(quiet), length(quiet) == 1L, !is.na(quiet))
  old <- withr::local_options(list(burrowlog.quiet = quiet), .local_envir = .local_envir)
  return(invisible(old))
}

#' @rdname local_quiet
#' @param code The code to evaluate.
#' @return `with_quiet()` returns the value of `code`.
#' @export
with_quiet <- function(quiet = TRUE, code) {
  local_quiet(quiet)
  return(force(code))
}

#' Use a fixed random seed temporarily
#'
#' Sets the seed and restores the previous random number generator state on
#' exit, so callers' random streams are left untouched.
#'
#' @param seed A single integer seed.
#' @param .local_envir The environment whose exit restores the previous state.
#' @return `local_burrow_seed()` returns `seed`, invisibly.
#' @export
local_burrow_seed <- function(seed, .local_envir = parent.frame()) {
  stopifnot(is.numeric(seed), length(seed) == 1L, !is.na(seed))
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = globalenv())

  # A session that never drew a random number has no seed to restore, so
  # remove the one set.seed() creates instead
  withr::defer(
    {
      if (had_seed) {
        assign(".Random.seed", old_seed, envir = globalenv())
      } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
        rm(".Random.seed", envir = globalenv())
      }
    },
    envir = .local_envir
  )

  set.seed(seed)
  return(invisible(seed))
}

#' @rdname local_burrow_seed
#' @param code The code to evaluate.
#' @return `with_burrow_seed()` returns the value of `code`.
#' @export
with_burrow_seed <- function(seed, code) {
  local_burrow_seed(seed)
  return(force(code))
}

#' Create a throwaway project directory
#'
#' Creates a temporary directory, writes `files` into it, and makes it the
#' working directory until the calling frame exits. Everything is deleted
#' afterwards.
#'
#' @param files A named list of file contents; names are relative paths and
#'   values are character vectors of lines.
#' @param .local_envir The environment whose exit restores the previous state.
#' @return `local_burrow_project()` returns the project path, invisibly.
#' @export
local_burrow_project <- function(files = list(), .local_envir = parent.frame()) {
  if (length(files) > 0L && (is.null(names(files)) || any(!nzchar(names(files))))) {
    rlang::abort(
      "`files` must be a named list of file contents.",
      class = c("burrowlog_error_files", "burrowlog_error")
    )
  }

  root <- withr::local_tempdir(pattern = "burrow-", .local_envir = .local_envir)
  for (rel_path in names(files)) {
    target <- file.path(root, rel_path)
    dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
    writeLines(files[[rel_path]], target)
  }

  # Registered after the tempdir, so it is undone first on exit
  withr::local_dir(root, .local_envir = .local_envir)
  return(invisible(root))
}

#' @rdname local_burrow_project
#' @param code The code to evaluate.
#' @return `with_burrow_project()` returns the value of `code`.
#' @export
with_burrow_project <- function(files = list(), code) {
  local_burrow_project(files)
  return(force(code))
}

#' Redirect the burrowlog log file temporarily
#'
#' Points the `burrowlog.log_file` option at a fresh temporary file and
#' removes the file when the calling frame exits.
#'
#' @param .local_envir The environment whose exit restores the previous state.
#' @return The path of the temporary log file, invisibly.
#' @keywords internal
local_log_file <- function(.local_envir = parent.frame()) {
  path <- withr::local_tempfile(fileext = ".log", .local_envir = .local_envir)
  file.create(path)
  withr::local_options(list(burrowlog.log_file = path), .local_envir = .local_envir)
  return(invisible(path))
}

#' Append a line to the burrowlog log file
#'
#' Writes a timestamped line to the file named by the `burrowlog.log_file`
#' option. Does nothing when the option is unset.
#'
#' @param level The severity level of the entry.
#' @param text The message text.
#' @return The log line, invisibly, or `NULL` when logging is disabled.
#' @export
log_line <- function(level = c("info", "warn", "error"), text) {
  level <- match.arg(level)
  path <- getOption("burrowlog.log_file")
  if (is.null(path)) {
    return(invisible(NULL))
  }

  stamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  line <- sprintf("%s [%s] %s", stamp, toupper(level), text)
  cat(line, "\n", file = path, append = TRUE, sep = "")
  return(invisible(line))
}

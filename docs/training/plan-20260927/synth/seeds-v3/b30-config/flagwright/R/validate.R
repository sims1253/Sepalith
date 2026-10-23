#' Check one parsed flag value
#'
#' Every kind of check first handles a missing value, then tests the shape
#' of `x` and stops with a message naming the flag.
#'
#' @param x The parsed value of the flag.
#' @param name The long name of the flag, used in error messages.
#' @param kind What `x` should be.
#' @param choices Accepted values when `kind` is `"choice"`.
#' @param allow_null Whether a missing (`NULL`) value is acceptable.
#' @return `x`, invisibly. Errors when the check fails.
#' @keywords internal
check_flag <- function(x, name,
                       kind = c("string", "count", "switch", "choice", "dir"),
                       choices = NULL, allow_null = FALSE) {
  kind <- match.arg(kind)
  if (is.null(x)) {
    if (allow_null) return(invisible(x))
    stop(sprintf("Flag `--%s` is required.", name), call. = FALSE)
  }

  switch(kind,
    string = {
      if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
        stop(sprintf("Flag `--%s` must be a single non-empty string.", name),
             call. = FALSE)
      }
    },
    count = {
      if (!is.numeric(x) || length(x) != 1L || is.na(x) || x < 0 || x != round(x)) {
        stop(sprintf("Flag `--%s` must be a single non-negative whole number.", name),
             call. = FALSE)
      }
    },
    switch = {
      if (!is.logical(x) || length(x) != 1L || is.na(x)) {
        stop(sprintf("Flag `--%s` must be a single TRUE or FALSE.", name),
             call. = FALSE)
      }
    },
    choice = {
      if (!is.character(x) || length(x) != 1L || !x %in% choices) {
        stop(sprintf("Flag `--%s` must be one of %s.", name,
                     paste0("\"", choices, "\"", collapse = ", ")),
             call. = FALSE)
      }
    },
    dir = {
      if (!is.character(x) || length(x) != 1L || !dir.exists(x)) {
        stop(sprintf("Flag `--%s` must name an existing directory.", name),
             call. = FALSE)
      }
    }
  )
  return(invisible(x))
}

#' Validate parsed backup arguments
#'
#' Runs one check per element of the list returned by
#' [parse_backup_args()]: `source`, `dest`, `keep`, `compress`, `dry_run`
#' and `verbose`. Finally checks that `source` and `dest` differ.
#'
#' @param opts A list as returned by [parse_backup_args()].
#' @return `opts`, invisibly.
#' @keywords internal
validate_backup_args <- function(opts) {
  # Validate `source`
  check_flag(opts$source, "source", "dir")

  # Validate `dest`
  check_flag(opts$dest, "dest", "string")
  if (!dir.exists(dirname(opts$dest))) {
    stop(sprintf("Flag `--%s` must name a directory whose parent exists.", "dest"),
         call. = FALSE)
  }

  # Validate `keep`
  check_flag(opts$keep, "keep", "count")
  if (opts$keep < 1L) {
    stop(sprintf("Flag `--%s` must be at least 1.", "keep"), call. = FALSE)
  }

  # Validate `compress`
  check_flag(opts$compress, "compress", "choice",
             choices = c("none", "gzip", "bzip2", "xz"))

  # Validate `dry_run`
  check_flag(opts$dry_run, "dry-run", "switch")

  # Validate `verbose`
  check_flag(opts$verbose, "verbose", "switch")

  if (normalizePath(opts$source, mustWork = FALSE) ==
      normalizePath(opts$dest, mustWork = FALSE)) {
    stop("Flags `--source` and `--dest` must point to different directories.",
         call. = FALSE)
  }
  return(invisible(opts))
}

#' Validate parsed report arguments
#'
#' Runs one check per element of the list returned by
#' [parse_report_args()]: `input`, `output`, `format`, `from`, `to` and
#' `quiet`. Finally checks that `from` is not after `to`.
#'
#' @param opts A list as returned by [parse_report_args()].
#' @return `opts`, invisibly.
#' @keywords internal
validate_report_args <- function(opts) {
  # Validate `input`
  check_flag(opts$input, "input", "string")
  if (!file.exists(opts$input)) {
    stop(sprintf("Flag `--%s` must name an existing file.", "input"),
         call. = FALSE)
  }

  # Validate `output`
  check_flag(opts$output, "output", "string", allow_null = TRUE)
  if (!is.null(opts$output) && !dir.exists(dirname(opts$output))) {
    stop(sprintf("Flag `--%s` must name a directory whose parent exists.", "output"),
         call. = FALSE)
  }

  # Validate `format`
  check_flag(opts$format, "format", "choice", choices = c("html", "pdf", "md"))

  # Validate `from`
  if (!is.null(opts$from) && !inherits(opts$from, "Date")) {
    stop(sprintf("Flag `--%s` must be a date.", "from"), call. = FALSE)
  }

  # Validate `to`
  if (!is.null(opts$to) && !inherits(opts$to, "Date")) {
    stop(sprintf("Flag `--%s` must be a date.", "to"), call. = FALSE)
  }

  # Validate `quiet`
  check_flag(opts$quiet, "quiet", "switch")

  if (!is.null(opts$from) && !is.null(opts$to) && opts$from > opts$to) {
    stop("Flag `--from` must not be after `--to`.", call. = FALSE)
  }
  return(invisible(opts))
}

#' Check numeric flags against bounds
#'
#' @inheritParams check_flag_relations
#' @param lower A named numeric vector of inclusive lower bounds.
#' @param upper A named numeric vector of inclusive upper bounds.
#' @keywords internal
check_flag_ranges <- function(opts, lower = numeric(), upper = numeric()) {
  for (name in names(lower)) {
    value <- opts[[name]]
    if (is.null(value)) next
    if (value < lower[[name]]) {
      stop(sprintf("Flag `--%s` must be at least %s, not %s.", name,
                   format(lower[[name]]), format(value)),
           call. = FALSE)
    }
  }

  for (name in names(upper)) {
    value <- opts[[name]]
    if (is.null(value)) next
    if (value > upper[[name]]) {
      stop(sprintf("Flag `--%s` must be at most %s, not %s.", name,
                   format(upper[[name]]), format(value)),
           call. = FALSE)
    }
  }
  return(invisible(opts))
}

#' Check conflicts and dependencies between flags
#'
#' @param opts A named list of parsed flag values.
#' @param conflicts A list of length-2 character vectors; each names two
#'   flags that cannot both be switched on.
#' @param requires A list of length-2 character vectors; the first flag can
#'   only be switched on together with the second.
#' @keywords internal
check_flag_relations <- function(opts, conflicts = list(), requires = list()) {
  is_on <- function(value) {
    !is.null(value) && !identical(value, FALSE)
  }

  for (pair in conflicts) {
    first <- pair[[1]]
    second <- pair[[2]]
    if (is_on(opts[[first]]) && is_on(opts[[second]])) {
      stop(sprintf("Flags `--%s` and `--%s` cannot be used together.",
                   first, second),
           call. = FALSE)
    }
  }

  for (pair in requires) {
    first <- pair[[1]]
    second <- pair[[2]]
    if (is_on(opts[[first]]) && !is_on(opts[[second]])) {
      stop(sprintf("Flag `--%s` can only be used together with `--%s`.",
                   first, second),
           call. = FALSE)
    }
  }
  return(invisible(opts))
}

#' Check that required flags are present
#'
#' @param opts A named list of parsed flag values.
#' @param required A character vector of flag names that must be set.
#' @keywords internal
check_required_flags <- function(opts, required) {
  missing_flags <- character()
  for (name in required) {
    if (is.null(opts[[name]])) {
      missing_flags <- c(missing_flags, name)
    }
  }

  if (length(missing_flags) > 0L) {
    stop(sprintf("Missing required flag(s): %s.",
                 paste0("`--", missing_flags, "`", collapse = ", ")),
         call. = FALSE)
  }
  return(invisible(opts))
}

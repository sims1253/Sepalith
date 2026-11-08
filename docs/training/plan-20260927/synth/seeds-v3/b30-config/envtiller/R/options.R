#' Default values of the package options
#'
#' @return A named list with elements `tiller.timeout`, `tiller.retries`,
#'   `tiller.verbose`, `tiller.cache.dir` and `tiller.log.level`.
#' @noRd
tiller.option.defaults <- function() {
  list(
    tiller.timeout = 60,
    tiller.retries = 3L,
    tiller.verbose = FALSE,
    tiller.cache.dir = file.path(tempdir(), "tiller-cache"),
    tiller.log.level = "info"
  )
}

#' Validate the value of one package option
#'
#' @param name Full option name.
#' @param value Proposed value.
#' @return The normalised value. Errors with a message naming the option
#'   when `value` is not acceptable.
#' @noRd
tiller.validate.option <- function(name = c("tiller.timeout", "tiller.retries",
                                            "tiller.verbose", "tiller.cache.dir",
                                            "tiller.log.level"),
                                   value) {
  name <- match.arg(name)
  switch(name,
    tiller.timeout = {
      if (!is.numeric(value) || length(value) != 1L || is.na(value) || value <= 0) {
        stop(sprintf("Option `%s` must be %s.", name, "a single positive number"),
             call. = FALSE)
      }
      as.numeric(value)
    },
    tiller.retries = {
      if (!is.numeric(value) || length(value) != 1L || is.na(value) || value < 0) {
        stop(sprintf("Option `%s` must be %s.", name, "a single non-negative number"),
             call. = FALSE)
      }
      as.integer(value)
    },
    tiller.verbose = {
      if (!is.logical(value) || length(value) != 1L || is.na(value)) {
        stop(sprintf("Option `%s` must be %s.", name, "a single TRUE or FALSE"),
             call. = FALSE)
      }
      value
    },
    tiller.cache.dir = {
      if (!is.character(value) || length(value) != 1L || !nzchar(value)) {
        stop(sprintf("Option `%s` must be %s.", name, "a non-empty path"),
             call. = FALSE)
      }
      path.expand(value)
    },
    tiller.log.level = {
      if (!is.character(value) || length(value) != 1L ||
          !value %in% c("debug", "info", "warn", "error")) {
        stop(sprintf("Option `%s` must be %s.", name, "debug, info, warn or error"),
             call. = FALSE)
      }
      value
    }
  )
}

#' Get the value of a package option
#'
#' @description
#' Falls back to the package default when the option is unset, and
#' validates whatever the user has set.
#'
#' @param name Option name, with or without the `tiller.` prefix.
#' @export
tiller.get.option <- function(name) {
  # Accept names with or without the prefix
  if (!startsWith(name, "tiller.")) {
    name <- paste0("tiller.", name)
  }

  defaults <- tiller.option.defaults()
  if (!name %in% names(defaults)) {
    stop(sprintf("Unknown option `%s`.", name), call. = FALSE)
  }
  value <- getOption(name, defaults[[name]])
  tiller.validate.option(name, value)
}

#' Set package options
#'
#' @param ... Named values, with or without the `tiller.` prefix.
#' @export
tiller.set.options <- function(...) {
  new <- list(...)
  if (length(new) == 0L) {
    return(invisible(list()))
  }
  if (is.null(names(new)) || any(!nzchar(names(new)))) {
    stop("All arguments must be named.", call. = FALSE)
  }

  # Accept names with or without the prefix
  full <- ifelse(startsWith(names(new), "tiller."), names(new),
                 paste0("tiller.", names(new)))
  unknown <- setdiff(full, names(tiller.option.defaults()))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown option(s): %s.", paste0("`", unknown, "`", collapse = ", ")),
         call. = FALSE)
  }

  for (i in seq_along(new)) {
    new[[i]] <- tiller.validate.option(full[[i]], new[[i]])
  }
  names(new) <- full
  invisible(options(new))
}

#' Reset package options to their defaults
#'
#' @description
#' Sets every option listed in the defaults, or only those that are still
#' unset when `only.missing` is `TRUE` (as done when the package loads).
#'
#' @param only.missing Whether to leave options the user already set.
#' @noRd
tiller.reset.options <- function(only.missing = FALSE) {
  defaults <- tiller.option.defaults()

  if (only.missing) {
    current <- options()[names(defaults)]
    defaults <- defaults[vapply(current, is.null, logical(1))]
  }
  if (length(defaults) == 0L) {
    return(invisible(list()))
  }
  invisible(options(defaults))
}

#' Set package options from environment variables
#'
#' @description
#' Reads one environment variable per option: `TILLER_TIMEOUT`,
#' `TILLER_RETRIES`, `TILLER_VERBOSE`, `TILLER_CACHE_DIR` and
#' `TILLER_LOG_LEVEL`. Variables that are unset or empty are skipped; the
#' rest are converted, validated and set together.
#'
#' @param env A named character vector of variables to use instead of the
#'   process environment, mainly for testing.
#' @return The previous values of the options that were set, invisibly.
#' @noRd
tiller.options.from.env <- function(env = Sys.getenv()) {
  new <- list()
  # tiller.timeout
  raw <- env["TILLER_TIMEOUT"]
  if (!is.na(raw) && nzchar(raw)) {
    new$tiller.timeout <- suppressWarnings(as.numeric(raw))
  }

  # tiller.retries
  raw <- env["TILLER_RETRIES"]
  if (!is.na(raw) && nzchar(raw)) {
    new$tiller.retries <- suppressWarnings(as.numeric(raw))
  }

  # tiller.verbose
  raw <- env["TILLER_VERBOSE"]
  if (!is.na(raw) && nzchar(raw)) {
    new$tiller.verbose <- as.logical(toupper(raw))
  }

  # tiller.cache.dir
  raw <- env["TILLER_CACHE_DIR"]
  if (!is.na(raw) && nzchar(raw)) {
    new$tiller.cache.dir <- unname(raw)
  }

  # tiller.log.level
  raw <- env["TILLER_LOG_LEVEL"]
  if (!is.na(raw) && nzchar(raw)) {
    new$tiller.log.level <- tolower(unname(raw))
  }

  if (length(new) == 0L) {
    return(invisible(list()))
  }
  for (name in names(new)) {
    new[[name]] <- tiller.validate.option(name, new[[name]])
  }
  invisible(options(new))
}

#' Tabulate the package options
#'
#' @return A data frame with one row per option and columns `option`,
#'   `value`, `default` and `is.default`; values are formatted as strings.
#' @export
tiller.options.table <- function() {
  defaults <- tiller.option.defaults()
  option <- names(defaults)
  value <- vapply(option, function(n) format(tiller.get.option(n)), character(1),
                  USE.NAMES = FALSE)
  default <- vapply(defaults, format, character(1), USE.NAMES = FALSE)

  data.frame(option = option, value = value, default = default,
             is.default = value == default, stringsAsFactors = FALSE)
}

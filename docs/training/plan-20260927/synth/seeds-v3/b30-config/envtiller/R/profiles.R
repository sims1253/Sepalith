#' Built-in settings for a deployment profile
#'
#' @param name Profile name.
#' @return A nested list with sections `server` (`host`, `port`, `debug`),
#'   `log` (`level`, `file`) and `cache` (`dir`, `max.mb`).
#' @export
profile.settings <- function(name = c("dev", "test", "staging", "prod")) {
  name <- match.arg(name)
  switch(name,
    dev = list(
      server = list(host = "127.0.0.1", port = 8080L, debug = TRUE),
      log = list(level = "debug", file = ""),
      cache = list(dir = file.path(tempdir(), "tiller-dev"), max.mb = 64)
    ),
    test = list(
      server = list(host = "127.0.0.1", port = 0L, debug = TRUE),
      log = list(level = "warn", file = ""),
      cache = list(dir = file.path(tempdir(), "tiller-test"), max.mb = 16)
    ),
    staging = list(
      server = list(host = "0.0.0.0", port = 8080L, debug = FALSE),
      log = list(level = "info", file = "tiller-staging.log"),
      cache = list(dir = "/var/cache/tiller-staging", max.mb = 512)
    ),
    prod = list(
      server = list(host = "0.0.0.0", port = 80L, debug = FALSE),
      log = list(level = "info", file = "tiller.log"),
      cache = list(dir = "/var/cache/tiller", max.mb = 2048)
    )
  )
}

#' Detect the active profile
#'
#' @param var Environment variable that names the profile.
#' @param default Profile used when the variable is unset.
#' @noRd
profile.detect <- function(var = "TILLER_PROFILE", default = "dev") {
  name <- tolower(Sys.getenv(var, unset = default))
  if (!nzchar(name)) {
    name <- default
  }
  if (identical(Sys.getenv("CI"), "true") && identical(name, default)) {
    name <- "test"
  }
  name
}

#' Resolve settings from all layers
#'
#' @details
#' Settings are resolved in four layers, each overriding the one before:
#' 1. the built-in profile from [profile.settings()];
#' 2. the user file `file`, a key=value file with dotted keys such as
#'    `server.port`;
#' 3. environment variables, see [env.override()];
#' 4. explicit `overrides` passed by the caller.
#'
#' @param profile Profile name.
#' @param file Optional path to a user settings file.
#' @param overrides A named list of dotted keys and values.
#' @param prefix Application prefix for environment variables.
#' @return A nested settings list with an attribute `sources`: a named
#'   character vector giving the layer each dotted key came from.
#' @export
profile.resolve <- function(profile = profile.detect(), file = NULL,
                            overrides = list(), prefix = "TILLER") {
  # 1. Built-in profile
  settings <- profile.settings(profile)
  keys <- unlist(lapply(names(settings), function(s) {
    paste(s, names(settings[[s]]), sep = ".")
  }))
  sources <- stats::setNames(rep("profile", length(keys)), keys)

  # 2. User file
  if (!is.null(file) && file.exists(file)) {
    lines <- trimws(readLines(file, warn = FALSE))
    lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
    for (line in lines) {
      key <- trimws(sub("=.*$", "", line))
      value <- type.convert(trimws(sub("^[^=]*=", "", line)), as.is = TRUE)
      section <- sub("\\..*$", "", key)
      settings[[section]][[sub("^[^.]*\\.", "", key)]] <- value
      sources[[key]] <- "file"
    }
  }

  # 3. Environment variables
  before <- unlist(settings)
  settings <- env.override(settings, prefix)
  after <- unlist(settings)
  changed <- names(after)[!mapply(identical, as.list(after), as.list(before[names(after)]))]
  sources[changed] <- "env"

  # 4. Explicit overrides
  for (key in names(overrides)) {
    section <- sub("\\..*$", "", key)
    settings[[section]][[sub("^[^.]*\\.", "", key)]] <- overrides[[key]]
    sources[[key]] <- "override"
  }
  attr(settings, "sources") <- sources
  settings
}

#' Check resolved settings
#'
#' @param settings A nested settings list as returned by [profile.resolve()].
#' @return A character vector of problems, one per invalid setting, checked
#'   in the order `server.host`, `server.port`, `server.debug`, `log.level`,
#'   `log.file`, `cache.dir`, `cache.max.mb`.
#' @noRd
profile.check <- function(settings) {
  problems <- character()
  bad <- function(key, what) sprintf("Setting `%s` must be %s.", key, what)

  # server.host
  host <- settings$server$host
  if (!is.character(host) || length(host) != 1L || !nzchar(host)) {
    problems <- c(problems, bad("server.host", "a non-empty string"))
  }

  # server.port
  port <- settings$server$port
  if (!is.numeric(port) || length(port) != 1L || port < 0 || port > 65535) {
    problems <- c(problems, bad("server.port", "a port number"))
  }

  # server.debug
  debug <- settings$server$debug
  if (!is.logical(debug) || length(debug) != 1L || is.na(debug)) {
    problems <- c(problems, bad("server.debug", "TRUE or FALSE"))
  }

  # log.level
  level <- settings$log$level
  if (!is.character(level) || !level %in% c("debug", "info", "warn", "error")) {
    problems <- c(problems, bad("log.level", "one of debug, info, warn, error"))
  }

  # log.file
  file <- settings$log$file
  if (!is.character(file) || length(file) != 1L) {
    problems <- c(problems, bad("log.file", "a string"))
  }

  # cache.dir
  dir <- settings$cache$dir
  if (!is.character(dir) || length(dir) != 1L || !nzchar(dir)) {
    problems <- c(problems, bad("cache.dir", "a non-empty string"))
  }

  # cache.max.mb
  max.mb <- settings$cache$max.mb
  if (!is.numeric(max.mb) || length(max.mb) != 1L || max.mb <= 0) {
    problems <- c(problems, bad("cache.max.mb", "a positive number"))
  }
  problems
}

#' Describe resolved settings
#'
#' @inheritParams profile.check
#' @return A character vector with one `key = value (source)` line per
#'   setting.
#' @noRd
profile.describe <- function(settings) {
  sources <- attr(settings, "sources")
  out <- character()

  for (section in names(settings)) {
    for (key in names(settings[[section]])) {
      dotted <- paste(section, key, sep = ".")
      value <- format(settings[[section]][[key]])
      src <- if (!is.null(sources) && dotted %in% names(sources)) {
        sources[[dotted]]
      } else {
        "unknown"
      }
      out <- c(out, sprintf("%s = %s (%s)", dotted, value, src))
    }
  }
  out
}

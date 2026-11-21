#' Build an environment variable name
#'
#' @description
#' Joins the parts with underscores, upper-cases them and replaces every
#' character that is not a letter, digit or underscore.
#'
#' @param prefix Application prefix, e.g. `"TILLER"`.
#' @param ... Further name parts such as a section and a key.
#' @return A single string such as `"TILLER_SERVER_PORT"`.
#' @noRd
env.var.name <- function(prefix, ...) {
  parts <- c(prefix, ...)
  name <- toupper(paste(parts, collapse = "_"))
  gsub("[^A-Z0-9_]", "_", name)
}

#' Read a typed environment variable
#'
#' @description
#' Returns `default` when the variable is unset or empty, and signals an
#' error naming the variable when its value cannot be converted.
#'
#' @param name Name of the environment variable.
#' @param type How to convert the value. Lists are comma-separated.
#' @param default Value returned when the variable is unset or empty.
#' @return A logical, integer, numeric or character value.
#' @noRd
env.get <- function(name, type = c("str", "flag", "int", "num", "list"),
                    default = NULL) {
  type <- match.arg(type)
  raw <- Sys.getenv(name, unset = "")
  if (!nzchar(raw)) {
    return(default)
  }

  switch(type,
    str = {
      value <- trimws(raw)
    },
    flag = {
      value <- switch(tolower(raw), `1` = , true = , yes = , on = TRUE,
                      `0` = , false = , no = , off = FALSE, NA)
      if (is.na(value)) {
        stop(sprintf("Environment variable %s=\"%s\" is not a valid %s.", name, raw, type),
             call. = FALSE)
      }
    },
    int = {
      value <- suppressWarnings(as.integer(raw))
      if (is.na(value) || !grepl("^-?[0-9]+$", raw)) {
        stop(sprintf("Environment variable %s=\"%s\" is not a valid %s.", name, raw, type),
             call. = FALSE)
      }
    },
    num = {
      value <- suppressWarnings(as.numeric(raw))
      if (is.na(value)) {
        stop(sprintf("Environment variable %s=\"%s\" is not a valid %s.", name, raw, type),
             call. = FALSE)
      }
    },
    list = {
      value <- trimws(strsplit(raw, ",", fixed = TRUE)[[1]])
      value <- value[nzchar(value)]
    }
  )
  value
}

#' Override settings from environment variables
#'
#' @description
#' Every setting has one variable named by [env.var.name()] from `prefix`,
#' the section and the key. A variable that is set replaces the value in
#' `config`; unset variables leave it alone.
#'
#' @param config A nested list with sections `server`, `log` and `cache`.
#' @param prefix Application prefix.
#' @details The settings read are `server.host`, `server.port`,
#'   `server.workers`, `server.debug`, `log.level`, `log.file`,
#'   `log.tags`, `cache.dir`, `cache.max.mb` and `cache.ttl`.
#' @return The updated `config`.
#' @export
env.override <- function(config, prefix = "TILLER") {
  # server.host
  config$server$host <- env.get(env.var.name(prefix, "server", "host"), "str",
                                default = config$server$host)
  # server.port
  config$server$port <- env.get(env.var.name(prefix, "server", "port"), "int",
                                default = config$server$port)
  # server.workers
  config$server$workers <- env.get(env.var.name(prefix, "server", "workers"), "int",
                                   default = config$server$workers)
  # server.debug
  config$server$debug <- env.get(env.var.name(prefix, "server", "debug"), "flag",
                                 default = config$server$debug)

  # log.level
  config$log$level <- env.get(env.var.name(prefix, "log", "level"), "str",
                              default = config$log$level)
  # log.file
  config$log$file <- env.get(env.var.name(prefix, "log", "file"), "str",
                             default = config$log$file)
  # log.tags
  config$log$tags <- env.get(env.var.name(prefix, "log", "tags"), "list",
                             default = config$log$tags)

  # cache.dir
  config$cache$dir <- env.get(env.var.name(prefix, "cache", "dir"), "str",
                              default = config$cache$dir)
  # cache.max.mb
  config$cache$max.mb <- env.get(env.var.name(prefix, "cache", "max.mb"), "num",
                                 default = config$cache$max.mb)
  # cache.ttl
  config$cache$ttl <- env.get(env.var.name(prefix, "cache", "ttl"), "num",
                              default = config$cache$ttl)
  config
}

#' Evaluate code with temporary environment variables
#'
#' @description
#' The current values of the variables named in `vars` are saved first,
#' remembering which were unset, and put back on exit even if `code` fails.
#'
#' @param vars A named character vector; `NA` values unset the variable.
#' @param code Code to evaluate.
#' @noRd
with.env.vars <- function(vars, code) {
  old <- Sys.getenv(names(vars), unset = NA, names = TRUE)
  on.exit({
    was.set <- old[!is.na(old)]
    if (length(was.set) > 0L) do.call(Sys.setenv, as.list(was.set))
    was.unset <- names(old)[is.na(old)]
    if (length(was.unset) > 0L) Sys.unsetenv(was.unset)
  }, add = TRUE)

  to.set <- vars[!is.na(vars)]
  if (length(to.set) > 0L) do.call(Sys.setenv, as.list(to.set))
  to.unset <- names(vars)[is.na(vars)]
  if (length(to.unset) > 0L) Sys.unsetenv(to.unset)
  force(code)
}

#' Report which settings come from the environment
#'
#' @param keys Dotted setting names such as `"server.port"`.
#' @param prefix Application prefix.
#' @param secret Regular expression matching variable names whose values
#'   must not be shown.
#' @return A data frame with columns `key`, `variable`, `set` and `value`;
#'   secret values are replaced by `"<hidden>"` and unset ones are `NA`.
#' @export
env.report <- function(keys, prefix = "TILLER",
                       secret = "PASSWORD|TOKEN|SECRET|KEY$") {
  variable <- vapply(keys, function(k) env.var.name(prefix, k), character(1),
                     USE.NAMES = FALSE)
  value <- Sys.getenv(variable, unset = NA, names = FALSE)

  set <- !is.na(value)
  hidden <- set & grepl(secret, variable)
  value[hidden] <- "<hidden>"
  data.frame(key = keys, variable = variable, set = set, value = value,
             stringsAsFactors = FALSE)
}

#' Find unrecognised environment variables
#'
#' @description
#' Lists variables that start with the prefix but do not correspond to any
#' known setting, which usually means a typo. For each, the closest known
#' variable is suggested.
#'
#' @inheritParams env.report
#' @return A named character vector: names are the unknown variables,
#'   values the suggested variables (`NA` when nothing is close).
#' @noRd
env.unused <- function(keys, prefix = "TILLER") {
  known <- vapply(keys, function(k) env.var.name(prefix, k), character(1),
                  USE.NAMES = FALSE)
  present <- names(Sys.getenv())
  present <- present[startsWith(present, paste0(toupper(prefix), "_"))]
  unknown <- setdiff(present, known)

  # Suggest the closest known variable for each unknown one
  suggestion <- vapply(unknown, function(v) {
    hit <- agrep(v, known, max.distance = 2, value = TRUE)
    if (length(hit) > 0L) hit[[1]] else NA_character_
  }, character(1))
  suggestion
}

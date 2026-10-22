#' Default configuration values
#'
#' @return A named list of default settings.
#' @keywords internal
config_defaults <- function() {
  list(
    base_url = "https://inventory.example.org/api/v2",
    token = NULL,
    timeout = 30,
    max_tries = 4L,
    cache_ttl = 3600,
    version = "0.3.1"
  )
}

#' Read configuration values from environment variables
#'
#' Variables are named `APICLERK_<FIELD>`; unset variables are omitted from the
#' result so that they do not override other sources.
#'
#' @param prefix Prefix of the environment variable names.
#' @return A named list of settings found in the environment.
#' @keywords internal
config_from_env <- function(prefix = "APICLERK_") {
  fields <- names(config_defaults())
  values <- Sys.getenv(paste0(prefix, toupper(fields)), unset = NA_character_)
  names(values) <- fields
  values[!is.na(values) & nzchar(values)] %>%
    as.list()
}

#' Read configuration values from a JSON file
#'
#' A missing file is not an error and yields an empty list.
#'
#' @param path Path to a JSON configuration file.
#' @return A named list of settings found in the file.
#' @keywords internal
config_from_file <- function(path) {
  if (is.null(path) || !file.exists(path)) {
    return(list())
  }
  parsed <- tryCatch(
    jsonlite::read_json(path, simplifyVector = TRUE),
    error = function(e) {
      stop("Could not parse config file ", path, ": ", conditionMessage(e), call. = FALSE)
    }
  )
  unknown <- setdiff(names(parsed), names(config_defaults()))
  if (length(unknown) > 0L) {
    warning("Ignoring unknown config field(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  parsed[intersect(names(parsed), names(config_defaults()))]
}

#' Build the active configuration
#'
#' Sources are merged in increasing order of priority: package defaults, the
#' config file, environment variables and finally explicit overrides.
#'
#' @param path Path to a JSON configuration file.
#' @param ... Named settings that override every other source.
#' @return A validated configuration list.
#' @export
clerk_config <- function(path = Sys.getenv("APICLERK_CONFIG", "~/.apiclerk.json"), ...) {
  overrides <- list(...)
  if (length(overrides) > 0L && (is.null(names(overrides)) || any(names(overrides) == ""))) {
    stop("All overrides passed to `clerk_config()` must be named.", call. = FALSE)
  }
  config_defaults() %>%
    utils::modifyList(config_from_file(path)) %>%
    utils::modifyList(config_from_env()) %>%
    utils::modifyList(overrides) %>%
    coerce_config() %>%
    validate_config()
}

#' Coerce configuration values to their expected types
#'
#' Values read from the environment arrive as strings; this converts them to
#' the types used by the defaults.
#'
#' @param config A configuration list.
#' @return The configuration list with coerced values.
#' @keywords internal
coerce_config <- function(config) {
  config$timeout <- as.numeric(config$timeout)
  config$max_tries <- as.integer(config$max_tries)
  config$cache_ttl <- as.numeric(config$cache_ttl)
  config$base_url <- sub("/+$", "", as.character(config$base_url))
  config
}

#' Validate a configuration list
#'
#' Collects every problem before failing so users can fix them in one go.
#'
#' @param config A configuration list.
#' @return `config`, unchanged, if it is valid.
#' @keywords internal
validate_config <- function(config) {
  problems <- character()
  if (!grepl("^https?://", config$base_url)) {
    problems <- c(problems, "`base_url` must start with http:// or https://")
  }
  if (is.na(config$timeout) || config$timeout <= 0) {
    problems <- c(problems, "`timeout` must be a positive number")
  }
  if (is.na(config$max_tries) || config$max_tries < 1L) {
    problems <- c(problems, "`max_tries` must be at least 1")
  }
  if (is.na(config$cache_ttl) || config$cache_ttl < 0) {
    problems <- c(problems, "`cache_ttl` must be zero or positive")
  }
  if (length(problems) > 0L) {
    stop("Invalid configuration:\n", paste0("* ", problems, collapse = "\n"), call. = FALSE)
  }
  config
}

#' Show the active configuration with secrets masked
#'
#' @param config A configuration list.
#' @return A data frame with one row per setting, invisibly.
#' @export
show_config <- function(config = clerk_config()) {
  if (!is.null(config$token)) {
    config$token <- paste0(substr(config$token, 1, 4), strrep("*", 8))
  }
  shown <- vapply(config, function(v) if (is.null(v)) "<unset>" else format(v), "")
  out <- data.frame(setting = names(shown), value = unname(shown))
  print(out, row.names = FALSE, right = FALSE)
  invisible(out)
}

#' Report where each configuration value comes from
#'
#' Resolves the configuration the same way as [clerk_config()] but records the
#' source that supplied each field, which helps when a setting seems to be
#' ignored.
#'
#' @param path Path to a JSON configuration file.
#' @param ... Named settings that override every other source.
#' @return A data frame with columns `setting`, `source` and `value`.
#' @export
config_sources <- function(path = Sys.getenv("APICLERK_CONFIG", "~/.apiclerk.json"), ...) {
  layers <- list(
    default = config_defaults(),
    file = config_from_file(path),
    env = config_from_env(),
    override = list(...)
  )
  fields <- names(layers$default)
  source <- rep("default", length(fields))
  value <- character(length(fields))
  names(source) <- names(value) <- fields
  for (layer in names(layers)) {
    present <- intersect(names(layers[[layer]]), fields)
    source[present] <- layer
  }
  for (field in fields) {
    v <- layers[[source[[field]]]][[field]]
    value[[field]] <- if (is.null(v)) {
      "<unset>"
    } else if (field == "token") {
      paste0(substr(v, 1, 4), strrep("*", 8))
    } else {
      format(v)
    }
  }
  data.frame(setting = fields, source = unname(source), value = unname(value)) %>%
    structure(path = path.expand(path))
}

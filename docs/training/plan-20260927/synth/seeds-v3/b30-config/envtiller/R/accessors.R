#' Create a validator for one setting type
#'
#' @param type Setting type.
#' @param choices Accepted values when `type` is `"choice"`.
#' @return A function of `(value, key)` that returns the normalised value or
#'   signals an error naming `key`.
#' @noRd
setting.validator <- function(type = c("int", "num", "flag", "str", "choice"),
                              choices = NULL) {
  type <- match.arg(type)
  switch(type,
    int = function(value, key) {
      out <- suppressWarnings(as.integer(value))
      if (length(out) != 1L || is.na(out)) {
        stop(sprintf("Setting `%s` must be a whole number.", key), call. = FALSE)
      }
      out
    },
    num = function(value, key) {
      out <- suppressWarnings(as.numeric(value))
      if (length(out) != 1L || is.na(out)) {
        stop(sprintf("Setting `%s` must be a number.", key), call. = FALSE)
      }
      out
    },
    flag = function(value, key) {
      out <- as.logical(value)
      if (length(out) != 1L || is.na(out)) {
        stop(sprintf("Setting `%s` must be TRUE or FALSE.", key), call. = FALSE)
      }
      out
    },
    str = function(value, key) {
      out <- as.character(value)
      if (length(out) != 1L || is.na(out)) {
        stop(sprintf("Setting `%s` must be a string.", key), call. = FALSE)
      }
      out
    },
    choice = function(value, key) {
      out <- as.character(value)
      if (length(out) != 1L || !out %in% choices) {
        stop(sprintf("Setting `%s` must be one of %s.", key,
                     paste(choices, collapse = ", ")), call. = FALSE)
      }
      out
    }
  )
}

#' Create an in-memory settings store
#'
#' @description
#' The store keeps its values in a closure. Every key has a declared type;
#' values are validated by [setting.validator()] on the way in.
#'
#' @param types A named character vector of setting types.
#' @param initial A named list of initial values.
#' @return A list of functions `get`, `set`, `has`, `unset`, `keys` and
#'   `as.list`.
#' @export
settings.store <- function(types, initial = list()) {
  validators <- lapply(types, setting.validator)
  values <- list()

  check.key <- function(key) {
    if (!key %in% names(types)) {
      stop(sprintf("Unknown setting `%s`.", key), call. = FALSE)
    }
  }
  get <- function(key, default = NULL) {
    check.key(key)
    if (key %in% names(values)) values[[key]] else default
  }
  set <- function(key, value) {
    check.key(key)
    old <- values[[key]]
    values[[key]] <<- validators[[key]](value, key)
    invisible(old)
  }
  has <- function(key) {
    check.key(key)
    key %in% names(values)
  }
  unset <- function(key) {
    check.key(key)
    old <- values[[key]]
    values[[key]] <<- NULL
    invisible(old)
  }
  keys <- function() {
    names(values)
  }
  as.list <- function() {
    values
  }

  for (key in names(initial)) {
    set(key, initial[[key]])
  }
  list(get = get, set = set, has = has, unset = unset, keys = keys,
       as.list = as.list)
}

#' Build a store from layered sources
#'
#' @details
#' Values are applied in four layers, each overriding the one before:
#' 1. `defaults`, a named list;
#' 2. the key=value file `file`, if it exists;
#' 3. environment variables named `<PREFIX>_<KEY>`, with dots in keys
#'    replaced by underscores;
#' 4. `args`, a named list of explicit values.
#' Every value passes through the validators of [settings.store()].
#'
#' @param types A named character vector of setting types.
#' @param defaults A named list of default values.
#' @param file Optional path to a key=value file.
#' @param prefix Application prefix for environment variables.
#' @param args A named list of explicit values.
#' @return A store as returned by [settings.store()], with one more
#'   function, `source()`, giving the layer each key came from.
#' @export
store.from.sources <- function(types, defaults = list(), file = NULL,
                               prefix = "TILLER", args = list()) {
  store <- settings.store(types)
  source <- character()

  # 1. Defaults
  for (key in names(defaults)) {
    store$set(key, defaults[[key]])
    source[[key]] <- "default"
  }

  # 2. File
  if (!is.null(file) && file.exists(file)) {
    lines <- trimws(readLines(file, warn = FALSE))
    lines <- lines[nzchar(lines) & !startsWith(lines, "#")]
    for (line in lines) {
      key <- trimws(sub("=.*$", "", line))
      store$set(key, trimws(sub("^[^=]*=", "", line)))
      source[[key]] <- "file"
    }
  }

  # 3. Environment variables
  for (key in names(types)) {
    var <- toupper(paste(prefix, gsub(".", "_", key, fixed = TRUE), sep = "_"))
    raw <- Sys.getenv(var, unset = NA)
    if (!is.na(raw)) {
      store$set(key, raw)
      source[[key]] <- "env"
    }
  }

  # 4. Explicit arguments
  for (key in names(args)) {
    store$set(key, args[[key]])
    source[[key]] <- "args"
  }
  store$source <- function() source
  store
}

#' Save and load a store as a key=value file
#'
#' @description
#' `store.save()` writes one `key = value` line per setting that is set,
#' after a header line naming the type of every setting. `store.load()`
#' reads such a file back, checks the header against the store and sets
#' every value through the store's validators.
#'
#' @param store A store created by [settings.store()].
#' @param path Path to the file.
#' @param types A named character vector of setting types.
#' @rdname store.save
#' @export
store.save <- function(store, path, types) {
  values <- store$as.list()
  header <- paste0("# types: ", paste(names(types), types, sep = ":", collapse = " "))
  lines <- character(length(values))
  for (i in seq_along(values)) {
    value <- format(values[[i]])
    if (grepl("\n", value)) {
      stop(sprintf("Setting `%s` contains a newline.", names(values)[[i]]),
           call. = FALSE)
    }
    lines[[i]] <- paste(names(values)[[i]], "=", value)
  }

  writeLines(c(header, lines), path)
  invisible(path)
}

#' @rdname store.save
#' @export
store.load <- function(store, path, types) {
  lines <- readLines(path, warn = FALSE)
  header <- paste0("# types: ", paste(names(types), types, sep = ":", collapse = " "))
  if (length(lines) == 0L || !identical(lines[[1]], header)) {
    stop(sprintf("File '%s' was not written for these settings.", path),
         call. = FALSE)
  }

  keys <- character(length(lines) - 1L)
  for (i in seq_along(keys)) {
    line <- lines[[i + 1L]]
    keys[[i]] <- trimws(sub("=.*$", "", line))
    store$set(keys[[i]], trimws(sub("^[^=]*=", "", line)))
  }
  invisible(keys)
}

#' Print the contents of a store
#'
#' @param store A store created by [settings.store()].
#' @param title Heading printed above the settings.
#' @importFrom cli cli_h3 cli_bullets
#' @export
store.show <- function(store, title = "Settings") {
  cli_h3(title)
  values <- store$as.list()

  if (length(values) == 0L) {
    cli_bullets(c(i = "No settings are set."))
    return(invisible(store))
  }
  lines <- vapply(names(values), function(key) {
    sprintf("{.field %s}: {.val %s}", key, format(values[[key]]))
  }, character(1))
  cli_bullets(stats::setNames(lines, rep("*", length(lines))))
  invisible(store)
}

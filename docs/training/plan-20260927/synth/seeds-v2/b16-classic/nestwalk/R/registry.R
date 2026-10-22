#' Create a registry
#'
#' A registry is an environment used as a mutable key-value store. Because
#' environments are not copied on modify, updates are visible to every
#' holder of the registry.
#'
#' @param ... Named initial entries.
#' @return An environment of class `nw_registry`.
#' @export
new_registry <- function(...) {
  reg <- new.env(parent = emptyenv())
  init <- list(...)
  if (length(init) > 0L) {
    if (is.null(names(init)) || any(names(init) == "")) {
      stop("initial registry entries must be named", call. = FALSE)
    }
    list2env(init, envir = reg)
  }

  class(reg) <- "nw_registry"
  return(reg)
}

#' Check that an object is a registry
#'
#' @param reg A registry created by [new_registry()].
#' @noRd
assert_registry <- function(reg) {
  if (!inherits(reg, "nw_registry")) {
    stop("'reg' must be a registry created by new_registry()", call. = FALSE)
  }
  return(invisible(reg))
}

#' Store a value in a registry
#'
#' @param reg A registry created by [new_registry()].
#' @param key A single string naming the entry.
#' @param value Value to store.
#' @return `reg`, invisibly, so calls can be chained.
#' @export
registry_set <- function(reg, key, value) {
  assert_registry(reg)
  # assign() would quietly use only the first of several keys
  if (!is.character(key) || length(key) != 1L || is.na(key) || !nzchar(key)) {
    stop("'key' must be a single non-empty string", call. = FALSE)
  }
  assign(key, value, envir = reg)
  return(invisible(reg))
}

#' Fetch a value from a registry
#'
#' @inheritParams registry_set
#' @param default Value returned when `key` is absent.
#' @return The stored value, or `default`.
#' @export
registry_get <- function(reg, key, default = NULL) {
  assert_registry(reg)
  if (!exists(key, envir = reg, inherits = FALSE)) {
    return(default)
  }
  return(get(key, envir = reg, inherits = FALSE))
}

#' Check whether a registry holds a key
#'
#' @inheritParams registry_set
#' @return `TRUE` or `FALSE`.
#' @export
registry_has <- function(reg, key) {
  assert_registry(reg)
  return(exists(key, envir = reg, inherits = FALSE))
}

#' Remove a value from a registry
#'
#' Removing an absent key is not an error.
#'
#' @inheritParams registry_set
#' @return `reg`, invisibly, so calls can be chained.
#' @export
registry_remove <- function(reg, key) {
  assert_registry(reg)
  if (exists(key, envir = reg, inherits = FALSE)) {
    rm(list = key, envir = reg)
  }
  return(invisible(reg))
}

#' Update a stored value in place
#'
#' Applies `f` to the current value (or `default` when absent) and stores
#' the result under the same key.
#'
#' @inheritParams registry_set
#' @param f Function of one argument returning the new value.
#' @param default Value passed to `f` when `key` is absent.
#' @return The new value, invisibly.
#' @export
registry_update <- function(reg, key, f, default = NULL) {
  current <- registry_get(reg, key, default = default)
  new_value <- f(current)
  registry_set(reg, key, new_value)
  return(invisible(new_value))
}

#' Take a snapshot of a registry
#'
#' @param reg A registry created by [new_registry()].
#' @param sorted If `TRUE`, entries are ordered by key.
#' @return A named list copy of the registry contents.
#' @export
registry_snapshot <- function(reg, sorted = TRUE) {
  assert_registry(reg)
  out <- as.list.environment(reg, all.names = TRUE)
  if (sorted) {
    out <- out[order(names(out))]
  }
  return(out)
}

#' Print a registry
#'
#' @param x A registry created by [new_registry()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.nw_registry <- function(x, ...) {
  keys <- sort(ls(x, all.names = TRUE))
  cat(sprintf("<nw_registry> %d entr%s\n", length(keys),
              if (length(keys) == 1L) "y" else "ies"))

  # One line per entry showing only the class, never the value itself
  for (k in keys) {
    cat(sprintf("  %s: %s\n", k, class(get(k, envir = x))[1]))
  }
  return(invisible(x))
}

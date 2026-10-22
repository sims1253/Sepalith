the_cache <- new.env(parent = emptyenv())

#' Build a cache key from arbitrary values
#'
#' Hashes the supplied values together with a namespace so that different
#' memoised functions never collide.
#'
#' @param namespace A string identifying the owner of the cache entry.
#' @param ... Values that determine the cached result.
#' @return A string key.
#' @keywords internal
cache_key <- function(namespace, ...) {
  check_string(namespace)
  values <- rlang::list2(...)
  paste0(namespace, "-", rlang::hash(values))
}

#' Check, read and write entries in the session cache
#'
#' The session cache is an environment that lives for the duration of the R
#' session. Entries expire after `ttl` seconds when a time-to-live is set.
#'
#' @param key A cache key, usually created by [cache_key()].
#' @return `cache_has()` returns `TRUE` or `FALSE`.
#' @rdname session-cache
#' @keywords internal
cache_has <- function(key) {
  if (!exists(key, envir = the_cache, inherits = FALSE)) {
    return(FALSE)
  }
  entry <- get(key, envir = the_cache, inherits = FALSE)
  if (is.null(entry$expires)) {
    return(TRUE)
  }
  expired <- Sys.time() > entry$expires
  if (expired) {
    rm(list = key, envir = the_cache)
  }
  !expired
}

#' @rdname session-cache
#' @param default The value returned when `key` is missing or expired.
#' @return `cache_get()` returns the cached value or `default`.
cache_get <- function(key, default = NULL) {
  if (!cache_has(key)) {
    return(default)
  }
  get(key, envir = the_cache, inherits = FALSE)$value
}

#' @rdname session-cache
#' @param value The value to store.
#' @param ttl Time-to-live in seconds, or `NULL` for entries that never expire.
#' @return `cache_set()` returns `value`, invisibly.
cache_set <- function(key, value, ttl = NULL) {
  expires <- NULL
  if (!is.null(ttl)) {
    check_number_between(ttl, min = 0)
    expires <- Sys.time() + ttl
  }
  assign(key, list(value = value, expires = expires), envir = the_cache)
  invisible(value)
}

#' Remove entries from the session cache
#'
#' @param namespace A string identifying the owner of the cache entries, or
#'   `NULL` to clear the whole cache.
#' @return The number of removed entries, invisibly.
#' @export
cache_clear <- function(namespace = NULL) {
  check_string(namespace, allow_null = TRUE)
  keys <- ls(the_cache, all.names = TRUE)
  if (!is.null(namespace)) {
    keys <- keys[startsWith(keys, paste0(namespace, "-"))]
  }
  rm(list = keys, envir = the_cache)
  if (tallyhook_is_verbose() && length(keys) > 0L) {
    cli::cli_inform("Removed {length(keys)} cached entr{?y/ies}.")
  }
  invisible(length(keys))
}

#' Memoise a function in the session cache
#'
#' Returns a wrapper around `f` that stores results keyed on its arguments.
#' Errors are never cached.
#'
#' @param f The function to memoise.
#' @param namespace A string identifying the owner of the cache entries.
#' @param ttl Time-to-live in seconds, or `NULL` for entries that never expire.
#' @return A function that accepts the same arguments as `f`.
#' @export
#' @examples
#' slow_square <- function(x) {
#'   Sys.sleep(1)
#'   x^2
#' }
#' fast_square <- memoise_tally(slow_square, "square")
#' fast_square(4)
#' fast_square(4)
memoise_tally <- function(f, namespace, ttl = NULL) {
  if (!is.function(f)) {
    cli::cli_abort(
      "{.arg f} must be a function, not {.obj_type_friendly {f}}.",
      class = c("tallyhook_error_function", "tallyhook_error_arg", "tallyhook_error")
    )
  }
  check_string(namespace)
  force(ttl)
  wrapper <- function(...) {
    key <- cache_key(namespace, ...)
    if (cache_has(key)) {
      return(cache_get(key))
    }
    result <- f(...)
    cache_set(key, result, ttl = ttl)
    result
  }
  wrapper
}

#' Read and write cached objects on disk
#'
#' Disk entries are stored as `.rds` files inside [tallyhook_cache_dir()].
#'
#' @param key A cache key, usually created by [cache_key()].
#' @return `disk_cache_read()` returns the stored object, or `NULL` when no
#'   file exists for `key`.
#' @rdname disk-cache
#' @keywords internal
disk_cache_read <- function(key) {
  path <- file.path(tallyhook_cache_dir(create = FALSE), paste0(key, ".rds"))
  if (!file.exists(path)) {
    return(NULL)
  }
  tryCatch(
    readRDS(path),
    error = function(cnd) {
      cli::cli_warn("Discarding unreadable cache file {.path {path}}.", parent = cnd)
      unlink(path)
      NULL
    }
  )
}

#' @rdname disk-cache
#' @param value The value to store.
#' @return `disk_cache_write()` returns the file path, invisibly.
disk_cache_write <- function(key, value) {
  dir <- tallyhook_cache_dir(create = TRUE)
  path <- file.path(dir, paste0(key, ".rds"))
  tmp <- tempfile(tmpdir = dir, fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(value, tmp)
  if (!file.rename(tmp, path)) {
    cli::cli_abort(
      "Could not write cache file {.path {path}}.",
      class = c("tallyhook_error_cache", "tallyhook_error")
    )
  }
  invisible(path)
}

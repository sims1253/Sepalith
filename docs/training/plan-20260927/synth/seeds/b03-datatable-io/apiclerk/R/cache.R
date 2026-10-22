#' Create a response cache backed by a directory
#'
#' Entries are stored as RDS files named by key and expire after `ttl`
#' seconds. The cache object is a plain list so it can be passed around freely.
#'
#' @param dir Directory holding cache files.
#' @param ttl Time to live of each entry, in seconds.
#' @return A list of class `clerk_cache`.
#' @export
new_cache <- function(dir = tools::R_user_dir("apiclerk", "cache"), ttl = 3600) {
  stopifnot(is.numeric(ttl), length(ttl) == 1L, ttl >= 0)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  return(structure(list(dir = dir, ttl = ttl), class = "clerk_cache"))
}

#' Compute a cache key for a request
#'
#' Query parameters are sorted by name first, so that the same request built in
#' a different order maps to the same key.
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters.
#' @return A single character string usable as a file name.
#' @keywords internal
cache_key <- function(path, query = list()) {
  query <- query[order(names(query))]
  parts <- c(path, paste0(names(query), "=", vapply(query, paste, "", collapse = ",")))
  raw_key <- paste(parts, collapse = "&")
  return(paste0(gsub("[^A-Za-z0-9]+", "_", path), "-", digest_string(raw_key)))
}

#' Hash a string into a short hexadecimal digest
#'
#' @param x A single character string.
#' @return An eight-character hexadecimal string.
#' @keywords internal
digest_string <- function(x) {
  tmp <- tempfile()
  on.exit(unlink(tmp), add = TRUE)
  writeLines(x, tmp, useBytes = TRUE)
  return(substr(unname(tools::md5sum(tmp)), 1L, 8L))
}

#' Check whether a fresh cache entry exists
#'
#' @param cache A `clerk_cache` object.
#' @param key A cache key as returned by `cache_key()`.
#' @return `TRUE` if the entry exists and has not expired.
#' @family cache accessors
#' @export
cache_has <- function(cache, key) {
  file <- file.path(cache$dir, paste0(key, ".rds"))
  if (!file.exists(file)) {
    return(FALSE)
  }
  age <- as.numeric(difftime(Sys.time(), file.mtime(file), units = "secs"))
  return(age <= cache$ttl)
}

#' Read an entry from the cache
#'
#' @param cache A `clerk_cache` object.
#' @param key A cache key as returned by `cache_key()`.
#' @param default Value returned when the entry is missing or expired.
#' @return The cached value, or `default`.
#' @family cache accessors
#' @export
cache_get <- function(cache, key, default = NULL) {
  if (!cache_has(cache, key)) {
    return(default)
  }
  file <- file.path(cache$dir, paste0(key, ".rds"))
  return(readRDS(file))
}

#' Write an entry to the cache
#'
#' @param cache A `clerk_cache` object.
#' @param key A cache key as returned by `cache_key()`.
#' @param value Any R object to store.
#' @return `value`, invisibly.
#' @family cache accessors
#' @export
cache_set <- function(cache, key, value) {
  file <- file.path(cache$dir, paste0(key, ".rds"))
  tmp <- paste0(file, ".tmp")
  saveRDS(value, tmp)
  file.rename(tmp, file)
  return(invisible(value))
}

#' Remove expired or all entries from the cache
#'
#' @param cache A `clerk_cache` object.
#' @param expired_only If `TRUE`, remove only entries older than the TTL.
#' @return The number of files removed, invisibly.
#' @family cache accessors
#' @export
cache_clear <- function(cache, expired_only = FALSE) {
  files <- list.files(cache$dir, pattern = "\\.rds$", full.names = TRUE)
  if (expired_only) {
    age <- as.numeric(difftime(Sys.time(), file.mtime(files), units = "secs"))
    files <- files[age > cache$ttl]
  }
  unlink(files)
  return(invisible(length(files)))
}

#' Perform a GET request, using the cache when possible
#'
#' On a cache hit no request is made. On a miss the response is fetched with
#' [clerk_get()] and stored before being returned.
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters.
#' @param cache A `clerk_cache` object.
#' @param refresh If `TRUE`, ignore any cached entry and fetch again.
#' @param config A configuration list as returned by [clerk_config()].
#' @return The parsed JSON body as a list.
#' @export
cached_get <- function(path, query = list(), cache = new_cache(), refresh = FALSE,
                       config = clerk_config()) {
  key <- cache_key(path, query)
  if (!refresh && cache_has(cache, key)) {
    return(cache_get(cache, key))
  }
  value <- clerk_get(path, query, config)
  cache_set(cache, key, value)
  return(value)
}

#' Print a cache object
#'
#' @param x A `clerk_cache` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.clerk_cache <- function(x, ...) {
  files <- list.files(x$dir, pattern = "\\.rds$", full.names = TRUE)
  size <- sum(file.size(files))
  cat("<clerk_cache>\n")
  cat("  dir:     ", x$dir, "\n", sep = "")
  cat("  ttl:     ", x$ttl, " s\n", sep = "")
  cat("  entries: ", length(files), " (", format(size, big.mark = ","), " bytes)\n", sep = "")
  return(invisible(x))
}

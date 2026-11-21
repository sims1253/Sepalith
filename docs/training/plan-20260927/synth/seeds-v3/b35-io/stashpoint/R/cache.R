#' @title Path of a cache entry
#' @param key Entry key: letters, digits, `_`, `-` and `.` only.
#' @param dir Cache directory.
#' @return The path of the entry's RDS file.
#' @keywords internal
cachePath = function(key, dir) {
  if (!is.character(key) || length(key) != 1 || !grepl("^[A-Za-z0-9_.-]+$", key)) {
    stop("`key` must be a single string of letters, digits, `_`, `-` or `.`.", call. = FALSE)
  }
  file.path(dir, paste0(key, ".rds"))
}

#' @title Read the cache index
#' @description The index lives in `index.dcf` inside the cache directory and
#'   holds one paragraph per entry with the fields `key`, `file`, `bytes`,
#'   `created`, `accessed` and `hits`.
#' @param dir Cache directory.
#' @return A data frame with one column per index field; zero rows when the
#'   index does not exist yet.
#' @keywords internal
cacheIndexRead = function(dir) {
  path = file.path(dir, "index.dcf")
  if (!file.exists(path)) {
    return(data.frame(key = character(), file = character(), bytes = numeric(),
                      created = character(), accessed = character(), hits = integer(),
                      stringsAsFactors = FALSE))
  }

  raw = read.dcf(path)
  absent = setdiff(c("key", "file", "bytes", "created", "accessed", "hits"), colnames(raw))
  if (length(absent) > 0) {
    stop(sprintf("Cache index is missing field(s): %s", paste(absent, collapse = ", ")),
         call. = FALSE)
  }
  index = data.frame(
    key = raw[, "key"],
    file = raw[, "file"],
    bytes = as.numeric(raw[, "bytes"]),
    created = raw[, "created"],
    accessed = raw[, "accessed"],
    hits = as.integer(raw[, "hits"]),
    stringsAsFactors = FALSE
  )
  rownames(index) = NULL
  index
}

#' @title Write the cache index
#' @param index A data frame with columns `key`, `file`, `bytes`, `created`,
#'   `accessed` and `hits`.
#' @param dir Cache directory.
#' @return The index path, invisibly. An empty index removes the file.
#' @keywords internal
cacheIndexWrite = function(index, dir) {
  path = file.path(dir, "index.dcf")
  if (nrow(index) == 0) {
    unlink(path)
    return(invisible(path))
  }
  absent = setdiff(c("key", "file", "bytes", "created", "accessed", "hits"), names(index))
  if (length(absent) > 0) {
    stop(sprintf("Cache index is missing field(s): %s", paste(absent, collapse = ", ")),
         call. = FALSE)
  }

  index = index[order(index$key), , drop = FALSE]
  out = data.frame(
    key = index$key,
    file = index$file,
    bytes = format(index$bytes, scientific = FALSE, trim = TRUE),
    created = index$created,
    accessed = index$accessed,
    hits = format(index$hits, trim = TRUE),
    stringsAsFactors = FALSE
  )

  write.dcf(out, path)
  invisible(path)
}

#' @title Store a value in the cache
#' @param key Entry key: letters, digits, `_`, `-` and `.` only.
#' @param value Any R object.
#' @param dir Cache directory, created if needed.
#' @return The entry path, invisibly.
#' @export
cachePut = function(key, value, dir) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  path = cachePath(key, dir)
  saveRDS(value, path)
  now = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")

  # Replace any existing index row for `key`
  index = cacheIndexRead(dir)
  index = index[index$key != key, , drop = FALSE]
  row = data.frame(
    key = key,
    file = basename(path),
    bytes = file.size(path),
    created = now,
    accessed = now,
    hits = 0L,
    stringsAsFactors = FALSE
  )
  cacheIndexWrite(rbind(index, row), dir)
  invisible(path)
}

#' @title Fetch a value from the cache
#' @description A hit increments the entry's `hits` and updates `accessed`.
#' @param key Entry key: letters, digits, `_`, `-` and `.` only.
#' @param dir Cache directory.
#' @param default Value returned when the entry does not exist.
#' @return The cached value or `default`.
#' @export
cacheGet = function(key, dir, default = NULL) {
  path = cachePath(key, dir)
  if (!file.exists(path)) {
    return(default)
  }
  value = readRDS(path)

  # Record the hit in the index row for `key`
  index = cacheIndexRead(dir)
  row = match(key, index$key)
  if (!is.na(row)) {
    index$accessed[row] = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
    index$hits[row] = index$hits[row] + 1L
    cacheIndexWrite(index, dir)
  }
  value
}

#' @title Drop a value from the cache
#' @param key Entry key: letters, digits, `_`, `-` and `.` only.
#' @param dir Cache directory.
#' @return `TRUE` if an entry was removed, `FALSE` otherwise, invisibly.
#' @export
cacheDrop = function(key, dir) {
  if (!dir.exists(dir)) {
    return(invisible(FALSE))
  }
  path = cachePath(key, dir)
  found = file.exists(path)
  unlink(path)

  # Remove the index row for `key`
  index = cacheIndexRead(dir)
  found = found || key %in% index$key
  index = index[index$key != key, , drop = FALSE]
  cacheIndexWrite(index, dir)
  invisible(found)
}

#' @title Shrink the cache to a size budget
#' @description Drops entries in the order given by `policy` until the total
#'   size of the remaining entries is at most `maxBytes`:
#'   `"oldest"` drops by `created`, `"stale"` by `accessed`, `"largest"` by
#'   decreasing `bytes` and `"unused"` by increasing `hits`. Ties are broken
#'   by `key`.
#' @param dir Cache directory.
#' @param maxBytes Size budget in bytes.
#' @param policy Order in which entries are dropped.
#' @return A character vector of the dropped keys.
#' @export
cachePrune = function(dir, maxBytes, policy = c("oldest", "stale", "largest", "unused")) {
  policy = match.arg(policy)
  if (!is.numeric(maxBytes) || length(maxBytes) != 1 || maxBytes < 0) {
    stop("`maxBytes` must be a single non-negative number.", call. = FALSE)
  }
  index = cacheIndexRead(dir)
  ord = switch(policy,
    oldest = {
      order(index$created, index$key)
    },
    stale = {
      order(index$accessed, index$key)
    },
    largest = {
      order(-index$bytes, index$key)
    },
    unused = {
      order(index$hits, index$key)
    }
  )
  index = index[ord, , drop = FALSE]

  dropped = character()
  total = sum(index$bytes)
  for (i in seq_len(nrow(index))) {
    if (total <= maxBytes) {
      break
    }
    cacheDrop(index$key[i], dir)
    dropped = c(dropped, index$key[i])
    total = total - index$bytes[i]
  }
  dropped
}

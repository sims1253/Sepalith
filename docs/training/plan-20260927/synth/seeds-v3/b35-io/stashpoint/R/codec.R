#' @title Compress a raw vector
#' @param x A raw vector.
#' @param method Compression method.
#' @return The compressed raw vector. Empty input is returned unchanged for
#'   every method.
#' @keywords internal
compressRaw = function(x, method = c("none", "gzip", "bzip2", "xz")) {
  method = match.arg(method)
  if (!is.raw(x)) {
    stop("`x` must be a raw vector.", call. = FALSE)
  }
  if (length(x) == 0) {
    return(x)
  }
  switch(method,
    none = {
      x
    },
    gzip = {
      memCompress(x, type = "gzip")
    },
    bzip2 = {
      memCompress(x, type = "bzip2")
    },
    xz = {
      memCompress(x, type = "xz")
    }
  )
}

#' @title Decompress a raw vector
#' @param x A raw vector produced by [compressRaw()].
#' @param method Compression method used for `x`.
#' @param size Expected decompressed length, or `NULL` to skip the check.
#' @return The decompressed raw vector.
#' @keywords internal
decompressRaw = function(x, method = c("none", "gzip", "bzip2", "xz"), size = NULL) {
  method = match.arg(method)
  if (!is.raw(x)) {
    stop("`x` must be a raw vector.", call. = FALSE)
  }
  if (length(x) == 0) {
    return(x)
  }
  out = switch(method,
    none = {
      x
    },
    gzip = {
      memDecompress(x, type = "gzip")
    },
    bzip2 = {
      memDecompress(x, type = "bzip2")
    },
    xz = {
      memDecompress(x, type = "xz")
    }
  )

  if (!is.null(size) && length(out) != size) {
    stop(sprintf("Expected %d bytes after decompression, got %d.", size, length(out)),
         call. = FALSE)
  }
  out
}

#' @title Serialise an object into a stash blob
#' @description A blob starts with a 14-byte header followed by the
#'   compressed serialisation of `object`. The header fields are, in order:
#'   `magic` (the four bytes `SPT1`), `version` (one byte, currently 1),
#'   `method` (one byte, the code of the compression method), `size` (four
#'   bytes, the uncompressed length) and `stored` (four bytes, the compressed
#'   length). Lengths are little-endian.
#' @param object Any R object.
#' @param method Compression method.
#' @return A raw vector.
#' @export
toBlob = function(object, method = c("none", "gzip", "bzip2", "xz")) {
  method = match.arg(method)
  codes = c(none = 0L, gzip = 1L, bzip2 = 2L, xz = 3L)
  payload = serialize(object, connection = NULL, version = 3)
  if (length(payload) > .Machine$integer.max) {
    stop("`object` is too large for a stash blob.", call. = FALSE)
  }
  body = compressRaw(payload, method)

  # Header field `magic`
  magic = charToRaw("SPT1")
  # Header field `version`
  version = as.raw(1L)
  # Header field `method`
  methodByte = as.raw(codes[[method]])
  # Header field `size`
  size = writeBin(length(payload), raw(), size = 4, endian = "little")
  # Header field `stored`
  stored = writeBin(length(body), raw(), size = 4, endian = "little")

  c(magic, version, methodByte, size, stored, body)
}

#' @title Deserialise a stash blob
#' @description Reads the header fields written by [toBlob()] in the same
#'   order and checks each of them before decompressing.
#' @param blob A raw vector produced by [toBlob()].
#' @return The stored object.
#' @export
fromBlob = function(blob) {
  codes = c(none = 0L, gzip = 1L, bzip2 = 2L, xz = 3L)
  if (!is.raw(blob) || length(blob) < 14) {
    stop("`blob` is too short to hold a stash header.", call. = FALSE)
  }

  # Header field `magic`
  if (!identical(blob[1:4], charToRaw("SPT1"))) {
    stop("`blob` does not start with the stash magic bytes.", call. = FALSE)
  }
  # Header field `version`
  version = as.integer(blob[5])
  if (version != 1L) {
    stop(sprintf("Unsupported stash version %d.", version), call. = FALSE)
  }
  # Header field `method`
  method = names(codes)[match(as.integer(blob[6]), codes)]
  if (is.na(method)) {
    stop(sprintf("Unknown compression code %d.", as.integer(blob[6])), call. = FALSE)
  }
  # Header field `size`
  size = readBin(blob[7:10], what = "integer", size = 4, endian = "little")
  # Header field `stored`
  stored = readBin(blob[11:14], what = "integer", size = 4, endian = "little")
  if (length(blob) - 14 != stored) {
    stop(sprintf("Expected %d stored bytes, found %d.", stored, length(blob) - 14),
         call. = FALSE)
  }

  payload = decompressRaw(blob[-(1:14)], method, size = size)
  unserialize(payload)
}

#' @title Write an object to a stash file
#' @param object Any R object.
#' @param path Destination file path.
#' @param method Compression method.
#' @param overwrite Whether an existing file may be replaced.
#' @return The number of bytes written, invisibly.
#' @export
writeBlob = function(object, path, method = c("none", "gzip", "bzip2", "xz"),
                     overwrite = TRUE) {
  method = match.arg(method)
  if (file.exists(path) && !overwrite) {
    stop(sprintf("Stash file `%s` already exists.", path), call. = FALSE)
  }

  blob = toBlob(object, method)
  tmp = paste0(path, ".part")
  con = file(tmp, open = "wb")
  writeBin(blob, con)
  close(con)
  if (!file.rename(tmp, path)) {
    unlink(tmp)
    stop(sprintf("Could not move `%s` to `%s`.", tmp, path), call. = FALSE)
  }
  invisible(length(blob))
}

#' @title Read an object from a stash file
#' @param path Path to a file written by [writeBlob()].
#' @return The stored object.
#' @export
readBlob = function(path) {
  size = file.size(path)
  if (is.na(size)) {
    stop(sprintf("Stash file `%s` does not exist.", path), call. = FALSE)
  }
  con = file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  blob = readBin(con, what = "raw", n = size)
  fromBlob(blob)
}

#' @title Compare compression methods on an object
#' @description Serialises `object` once and compresses it with each method,
#'   so callers can pick the smallest.
#' @param object Any R object.
#' @param methods Compression methods to try.
#' @return A data frame with columns `method`, `bytes`, `ratio` (compressed
#'   size over uncompressed size) and `seconds` (time to compress and
#'   decompress), one row per method, smallest first.
#' @export
compareMethods = function(object, methods = c("none", "gzip", "bzip2", "xz")) {
  methods = match.arg(methods, several.ok = TRUE)
  payload = serialize(object, connection = NULL, version = 3)
  bytes = numeric(length(methods))
  seconds = numeric(length(methods))
  for (i in seq_along(methods)) {
    started = proc.time()[["elapsed"]]
    packed = compressRaw(payload, methods[i])
    if (!identical(decompressRaw(packed, methods[i]), payload)) {
      stop(sprintf("Method `%s` did not round-trip.", methods[i]), call. = FALSE)
    }
    seconds[i] = proc.time()[["elapsed"]] - started
    bytes[i] = length(packed)
  }

  result = data.frame(method = methods, bytes = bytes, ratio = bytes / length(payload),
                      seconds = seconds, stringsAsFactors = FALSE)
  result = result[order(result$bytes), , drop = FALSE]
  rownames(result) = NULL
  result
}

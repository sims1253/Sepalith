#' Encode tile indices as text keys
#'
#' @param x,y Integer tile indices.
#' @param zoom Zoom level, recycled to the length of `x`.
#' @param format One of `"quadkey"`, `"path"`, `"tms_path"`, `"morton"` or
#'   `"hex"`.
#' @return A character vector of keys.
#' @export
encodeTileKey = function(x, y, zoom, format = c("quadkey", "path", "tms_path", "morton",
                                                "hex")) {
  format = match.arg(format)
  zoom = rep_len(zoom, length(x))
  keys = switch(format,
    quadkey = {
      vapply(seq_along(x), function(i) {
        bits = rev(seq_len(zoom[i]) - 1)
        digits = (x[i] %/% 2^bits) %% 2 + 2 * ((y[i] %/% 2^bits) %% 2)
        paste(digits, collapse = "")
      }, character(1))
    },
    path = {
      sprintf("%d/%d/%d", zoom, x, y)
    },
    tms_path = {
      # TMS counts rows from the south
      sprintf("%d/%d/%d", zoom, x, 2^zoom - 1 - y)
    },
    morton = {
      vapply(seq_along(x), function(i) {
        bits = rev(seq_len(zoom[i]) - 1)
        digits = (x[i] %/% 2^bits) %% 2 + 2 * ((y[i] %/% 2^bits) %% 2)
        sprintf("%d:%.0f", zoom[i], sum(digits * 4^rev(seq_along(digits) - 1)))
      }, character(1))
    },
    hex = {
      sprintf("%02x%06x%06x", as.integer(zoom), as.integer(x), as.integer(y))
    }
  )
  return(keys)
}

#' Decode text keys back to tile indices
#'
#' @param key A character vector of keys.
#' @param format One of `"quadkey"`, `"path"`, `"tms_path"`, `"morton"` or
#'   `"hex"`.
#' @return A data frame with integer columns `x`, `y` and `zoom`.
#' @export
decodeTileKey = function(key, format = c("quadkey", "path", "tms_path", "morton", "hex")) {
  format = match.arg(format)
  out = switch(format,
    quadkey = {
      parts = lapply(strsplit(key, ""), as.integer)
      zoom = lengths(parts)
      x = vapply(parts, function(d) sum((d %% 2) * 2^rev(seq_along(d) - 1)), numeric(1))
      y = vapply(parts, function(d) sum((d %/% 2) * 2^rev(seq_along(d) - 1)), numeric(1))
      data.frame(x = x, y = y, zoom = zoom)
    },
    path = {
      parts = do.call(rbind, lapply(strsplit(key, "/", fixed = TRUE), as.numeric))
      zoom = parts[, 1]
      x = parts[, 2]
      y = parts[, 3]
      data.frame(x = x, y = y, zoom = zoom)
    },
    tms_path = {
      parts = do.call(rbind, lapply(strsplit(key, "/", fixed = TRUE), as.numeric))
      zoom = parts[, 1]
      x = parts[, 2]
      y = 2^zoom - 1 - parts[, 3]
      data.frame(x = x, y = y, zoom = zoom)
    },
    morton = {
      parts = do.call(rbind, lapply(strsplit(key, ":", fixed = TRUE), as.numeric))
      zoom = parts[, 1]
      digits = lapply(seq_along(zoom), function(i) {
        (parts[i, 2] %/% 4^rev(seq_len(zoom[i]) - 1)) %% 4
      })
      x = vapply(digits, function(d) sum((d %% 2) * 2^rev(seq_along(d) - 1)), numeric(1))
      y = vapply(digits, function(d) sum((d %/% 2) * 2^rev(seq_along(d) - 1)), numeric(1))
      data.frame(x = x, y = y, zoom = zoom)
    },
    hex = {
      zoom = strtoi(substr(key, 1, 2), 16L)
      x = strtoi(substr(key, 3, 8), 16L)
      y = strtoi(substr(key, 9, 14), 16L)
      data.frame(x = x, y = y, zoom = zoom)
    }
  )
  out[] = lapply(out, as.integer)
  return(out)
}

#' Check that keys are well formed
#'
#' @param key A character vector of keys.
#' @param format One of `"quadkey"`, `"path"`, `"tms_path"`, `"morton"` or
#'   `"hex"`.
#' @param error Whether to stop on the first bad key instead of returning
#'   a logical vector.
#' @return A logical vector, or `key` invisibly when `error = TRUE` and all
#'   keys are valid.
#' @export
isTileKey = function(key, format = c("quadkey", "path", "tms_path", "morton", "hex"),
                     error = FALSE) {
  format = match.arg(format)
  ok = switch(format,
    quadkey = {
      pattern = "^[0-3]{1,24}$"
      grepl(pattern, key)
    },
    path = {
      pattern = "^[0-9]{1,2}/[0-9]+/[0-9]+$"
      grepl(pattern, key)
    },
    tms_path = {
      pattern = "^[0-9]{1,2}/[0-9]+/[0-9]+$"
      grepl(pattern, key)
    },
    morton = {
      pattern = "^[0-9]{1,2}:[0-9]+$"
      grepl(pattern, key)
    },
    hex = {
      pattern = "^[0-9a-f]{14}$"
      grepl(pattern, key)
    }
  )
  ok[is.na(key)] = FALSE
  if (!error) {
    return(ok)
  }
  if (!all(ok)) {
    bad = key[!ok][1]
    stop(sprintf("\"%s\" is not a valid %s key.", bad, format), call. = FALSE)
  }
  return(invisible(key))
}

#' Change a key from one format to another
#'
#' @param key A character vector of keys.
#' @param from,to Key formats accepted by [encodeTileKey()].
#' @return A character vector of keys in the `to` format.
#' @export
convertTileKey = function(key, from, to) {
  formats = c("quadkey", "path", "tms_path", "morton", "hex")
  from = match.arg(from, formats)
  to = match.arg(to, formats)
  if (from == to) {
    return(key)
  }
  isTileKey(key, from, error = TRUE)
  tiles = decodeTileKey(key, from)
  return(encodeTileKey(tiles$x, tiles$y, tiles$zoom, to))
}

#' Describe a vector of keys
#'
#' @param key A character vector of keys.
#' @param format Key format accepted by [decodeTileKey()].
#' @return A data frame with columns `key`, `x`, `y`, `zoom`, `parent`,
#'   `quadkey` and `path`, where `parent`, `quadkey` and `path` are keys in
#'   their own formats (`parent` in `format`).
#' @export
describeTileKeys = function(key, format = "quadkey") {
  isTileKey(key, format, error = TRUE)
  tiles = decodeTileKey(key, format)
  out = data.frame(key = key, stringsAsFactors = FALSE)

  # Columns x, y, zoom
  out$x = tiles$x
  out$y = tiles$y
  out$zoom = tiles$zoom

  # Column parent
  hasParent = tiles$zoom > 0
  out$parent = NA_character_
  out$parent[hasParent] = encodeTileKey(tiles$x[hasParent] %/% 2,
    tiles$y[hasParent] %/% 2, tiles$zoom[hasParent] - 1, format)

  # Column quadkey
  out$quadkey = encodeTileKey(tiles$x, tiles$y, tiles$zoom, "quadkey")

  # Column path
  out$path = encodeTileKey(tiles$x, tiles$y, tiles$zoom, "path")
  return(out)
}

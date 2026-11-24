#' Encode an opaque pagination cursor
#'
#' Each field is written as `key=value`, the fields are joined with `;` and
#' the result is hex encoded so that clients treat it as opaque.
#'
#' @param id Integer id of the last item on the page.
#' @param createdAt `POSIXct` creation time of the last item.
#' @param direction `"after"` or `"before"`.
#' @param sortKey Name of the field the listing is sorted by.
#' @param version Cursor format version.
#' @return A single string of hex digits.
#' @export
encodeCursor = function(id, createdAt, direction = "after", sortKey = "created_at",
                        version = 1L) {
  fields = character(0)
  # Field `id`
  if (!is.numeric(id) || length(id) != 1L || is.na(id)) {
    stop(sprintf("Cursor field `%s` is invalid.", "id"), call. = FALSE)
  }
  fields[["id"]] = format(as.integer(id))
  # Field `createdAt`
  if (!inherits(createdAt, "POSIXct") || length(createdAt) != 1L || is.na(createdAt)) {
    stop(sprintf("Cursor field `%s` is invalid.", "createdAt"), call. = FALSE)
  }
  fields[["createdAt"]] = format(as.numeric(createdAt), nsmall = 3)
  # Field `direction`
  if (!direction %in% c("after", "before")) {
    stop(sprintf("Cursor field `%s` is invalid.", "direction"), call. = FALSE)
  }
  fields[["direction"]] = direction
  # Field `sortKey`
  if (!is.character(sortKey) || length(sortKey) != 1L || grepl("[=;]", sortKey)) {
    stop(sprintf("Cursor field `%s` is invalid.", "sortKey"), call. = FALSE)
  }
  fields[["sortKey"]] = sortKey
  # Field `version`
  if (!is.numeric(version) || length(version) != 1L || version < 1) {
    stop(sprintf("Cursor field `%s` is invalid.", "version"), call. = FALSE)
  }
  fields[["version"]] = format(as.integer(version))
  plain = paste(paste0(names(fields), "=", fields), collapse = ";")
  return(paste(as.character(charToRaw(plain)), collapse = ""))
}

#' Decode an opaque pagination cursor
#'
#' The inverse of [encodeCursor()].
#'
#' @param cursor A single string of hex digits.
#' @return A list with elements `id`, `createdAt`, `direction`, `sortKey`
#'   and `version`.
#' @export
decodeCursor = function(cursor) {
  if (!is.character(cursor) || length(cursor) != 1L || !grepl("^([0-9a-f]{2})+$", cursor)) {
    stop("`cursor` is not a valid cursor string.", call. = FALSE)
  }
  starts = seq(1L, nchar(cursor), by = 2L)
  bytes = as.raw(strtoi(substring(cursor, starts, starts + 1L), base = 16L))
  parts = strsplit(strsplit(rawToChar(bytes), ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  fields = stats::setNames(vapply(parts, `[`, "", 2L), vapply(parts, `[`, "", 1L))
  out = list()
  # Field `id`
  if (is.na(fields["id"])) {
    stop(sprintf("Cursor field `%s` is missing.", "id"), call. = FALSE)
  }

  out$id = as.integer(fields[["id"]])
  # Field `createdAt`
  if (is.na(fields["createdAt"])) {
    stop(sprintf("Cursor field `%s` is missing.", "createdAt"), call. = FALSE)
  }

  out$createdAt = as.POSIXct(as.numeric(fields[["createdAt"]]), origin = "1970-01-01",
    tz = "UTC")
  # Field `direction`
  if (is.na(fields["direction"])) {
    stop(sprintf("Cursor field `%s` is missing.", "direction"), call. = FALSE)
  }

  out$direction = fields[["direction"]]
  # Field `sortKey`
  if (is.na(fields["sortKey"])) {
    stop(sprintf("Cursor field `%s` is missing.", "sortKey"), call. = FALSE)
  }

  out$sortKey = fields[["sortKey"]]
  # Field `version`
  if (is.na(fields["version"])) {
    stop(sprintf("Cursor field `%s` is missing.", "version"), call. = FALSE)
  }

  out$version = as.integer(fields[["version"]])
  return(out)
}

#' Select the items on one side of a cursor
#'
#' Items are ordered by `created_at`, then `id`. The page holds up to
#' `limit` items strictly after or strictly before the cursor position.
#'
#' @param items A data frame with columns `id` and `created_at` (POSIXct).
#' @param cursor A decoded cursor from [decodeCursor()], or `NULL` for the
#'   first page.
#' @param limit Maximum number of items to return.
#' @param direction `"after"` or `"before"`; ignored when `cursor` is
#'   supplied, which carries its own direction.
#' @return A list with elements `items` (the page, in listing order),
#'   `nextCursor` and `prevCursor` (encoded cursors, or `NA` at the ends).
#' @export
cursorWindow = function(items, cursor = NULL, limit = 20L, direction = "after") {
  items = items[order(items$created_at, items$id), , drop = FALSE]
  n = nrow(items)
  if (!is.null(cursor)) {
    direction = cursor$direction
    key = as.numeric(items$created_at)
    at = as.numeric(cursor$createdAt)
    isAfter = key > at | (key == at & items$id > cursor$id)
    isBefore = key < at | (key == at & items$id < cursor$id)
  } else {
    isAfter = rep(TRUE, n)
    isBefore = rep(TRUE, n)
  }
  if (direction == "after") {
    # Take the first `limit` rows after the cursor
    rows = utils::head(which(isAfter), limit)
    hasNext = length(rows) > 0L && max(rows) < n
    hasPrev = !is.null(cursor) && any(!isAfter)
  } else {
    # Take the last `limit` rows before the cursor
    rows = utils::tail(which(isBefore), limit)
    hasPrev = length(rows) > 0L && min(rows) > 1L
    hasNext = !is.null(cursor) && any(!isBefore)
  }
  page = items[rows, , drop = FALSE]
  nextCursor = NA_character_
  prevCursor = NA_character_
  if (hasNext) {
    last = page[nrow(page), ]
    nextCursor = encodeCursor(last$id, last$created_at, "after")
  }
  if (hasPrev && nrow(page) > 0L) {
    first = page[1L, ]
    prevCursor = encodeCursor(first$id, first$created_at, "before")
  }
  rownames(page) = NULL
  return(list(items = page, nextCursor = nextCursor, prevCursor = prevCursor))
}

#' Build a cursor from the last item of a parsed page
#'
#' @param items A list of items, each a nested list as parsed from JSON.
#' @param idPath Path to the item id, as a character vector of names.
#' @param timePath Path to the item creation time (ISO 8601 text).
#' @param direction `"after"` or `"before"`.
#' @return An encoded cursor, or `NA` when `items` is empty.
#' @export
cursorFromItems = function(items, idPath = "id", timePath = "created_at",
                           direction = c("after", "before")) {
  direction = match.arg(direction)
  if (length(items) == 0L) {
    return(NA_character_)
  }

  # Use the last item when paging forward and the first when paging back
  item = if (direction == "after") items[[length(items)]] else items[[1L]]
  # Path `idPath`
  id = item
  for (key in idPath) {
    id = id[[key]]
    if (is.null(id)) {
      stop(sprintf("Item has no value at `%s`.", paste(idPath, collapse = "$")), call. = FALSE)
    }
  }

  # Path `timePath`
  createdAt = item
  for (key in timePath) {
    createdAt = createdAt[[key]]
    if (is.null(createdAt)) {
      stop(sprintf("Item has no value at `%s`.", paste(timePath, collapse = "$")), call. = FALSE)
    }
  }
  createdAt = as.POSIXct(sub("Z$", "", createdAt), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  return(encodeCursor(id, createdAt, direction))
}

#' Describe a decoded cursor
#'
#' @param cursor A decoded cursor from [decodeCursor()].
#' @return A single string such as
#'   `"after id 12 at 2024-01-02 03:04:05 (by created_at, v1)"`.
#' @export
formatCursor = function(cursor) {
  parts = character(0)
  # Part `direction`
  if (!is.null(cursor$direction)) {
    parts = c(parts, cursor$direction)
  }

  # Part `id`
  if (!is.null(cursor$id)) {
    parts = c(parts, paste("id", cursor$id))
  }

  # Part `createdAt`
  if (!is.null(cursor$createdAt)) {
    parts = c(parts, paste("at", format(cursor$createdAt, "%Y-%m-%d %H:%M:%S", tz = "UTC")))
  }

  # Part `sortKey` and `version`
  if (!is.null(cursor$sortKey) && !is.null(cursor$version)) {
    parts = c(parts, sprintf("(by %s, v%d)", cursor$sortKey, cursor$version))
  }

  return(paste(parts, collapse = " "))
}

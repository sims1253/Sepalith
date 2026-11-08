#' Unwrap the item list from a response envelope
#'
#' @description
#' APIs wrap their results differently; each envelope style names where the
#' items and the total count live.
#'
#' @param body Parsed response body.
#' @param envelope One of `"data"` (`data`, `meta.total`), `"results"`
#'   (`results`, `count`), `"items"` (`items`, `total_count`), `"value"`
#'   (`value`, `@odata.count`) or `"none"` (the body is the item list).
#' @return A list with elements `items` and `total` (`NA` when not reported).
#' @keywords internal
unwrap.envelope <- function(body,
                            envelope = c("data", "results", "items", "value", "none")) {
  envelope <- match.arg(envelope)
  count.or.na <- function(x) if (is.null(x)) NA_integer_ else as.integer(x)

  switch(envelope,
    data = {
      items <- body[["data"]]
      total <- count.or.na(body[["meta"]][["total"]])
      list(items = if (is.null(items)) list() else items, total = total)
    },
    results = {
      items <- body[["results"]]
      total <- count.or.na(body[["count"]])
      list(items = if (is.null(items)) list() else items, total = total)
    },
    items = {
      items <- body[["items"]]
      total <- count.or.na(body[["total_count"]])
      list(items = if (is.null(items)) list() else items, total = total)
    },
    value = {
      items <- body[["value"]]
      total <- count.or.na(body[["@odata.count"]])
      list(items = if (is.null(items)) list() else items, total = total)
    },
    none = {
      items <- if (is.null(names(body))) body else list(body)
      list(items = items, total = length(items))
    }
  )
}

#' Flatten one nested record
#'
#' @description
#' Named sub-lists become `parent.child` columns; unnamed lists of scalars
#' are collapsed with `collapse`; empty values become `NA`.
#'
#' @param x One record, a nested list.
#' @param sep Separator between parent and child names.
#' @param collapse Separator for collapsed unnamed lists.
#' @param prefix Name prefix used in recursive calls.
#' @return A named list of length-one atomic values.
#' @keywords internal
flatten.record <- function(x, sep = ".", collapse = ",", prefix = NULL) {
  out <- list()
  for (nm in names(x)) {
    key <- if (is.null(prefix)) nm else paste(prefix, nm, sep = sep)
    value <- x[[nm]]
    if (is.null(value) || length(value) == 0L) {
      # Empty: keep the column as NA
      out[[key]] <- NA
    } else if (is.list(value) && !is.null(names(value))) {
      # Named list: recurse with the key as prefix
      out <- c(out, flatten.record(value, sep = sep, collapse = collapse, prefix = key))
    } else if (is.list(value)) {
      # Unnamed list: collapse scalars, keep a count for objects
      scalars <- vapply(value, function(v) is.atomic(v) && length(v) == 1L, logical(1))
      out[[key]] <- if (all(scalars)) {
        paste(unlist(value), collapse = collapse)
      } else {
        length(value)
      }
    } else if (length(value) > 1L) {
      out[[key]] <- paste(value, collapse = collapse)
    } else {
      out[[key]] <- value
    }
  }

  out
}

#' Convert a list of records to a data frame
#'
#' @param items A list of records, each a nested list.
#' @param sep Separator between parent and child names.
#' @param collapse Separator for collapsed unnamed lists.
#' @return A data frame with one row per item and the union of all flattened
#'   columns, in order of first appearance.
#' @export
records.to.frame <- function(items, sep = ".", collapse = ",") {
  if (length(items) == 0L) {
    return(data.frame())
  }
  flat <- lapply(items, flatten.record, sep = sep, collapse = collapse)
  columns <- unique(unlist(lapply(flat, names), use.names = FALSE))
  out <- data.frame(row.names = seq_along(flat))
  for (col in columns) {
    values <- lapply(flat, function(r) if (is.null(r[[col]])) NA else r[[col]])
    out[[col]] <- unlist(values, use.names = FALSE)
  }

  out
}

#' Coerce a parsed column to a type
#'
#' @param x An atomic vector from [records.to.frame()].
#' @param type One of `"auto"`, `"integer"`, `"double"`, `"logical"`,
#'   `"datetime"` or `"date"`. `"auto"` picks the narrowest type that keeps
#'   every non-missing value.
#' @return The converted vector.
#' @export
coerce.column <- function(x,
                          type = c("auto", "integer", "double", "logical", "datetime",
                                   "date")) {
  type <- match.arg(type)
  text <- as.character(x)
  if (type == "auto") {
    present <- text[!is.na(text)]
    type <- if (length(present) == 0L) {
      "logical"
    } else if (all(present %in% c("TRUE", "FALSE", "true", "false"))) {
      "logical"
    } else if (all(grepl("^-?[0-9]+$", present))) {
      "integer"
    } else if (!anyNA(suppressWarnings(as.numeric(present)))) {
      "double"
    } else if (all(grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}T", present))) {
      "datetime"
    } else if (all(grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", present))) {
      "date"
    } else {
      return(text)
    }
  }
  switch(type,
    integer = as.integer(text),
    double = as.numeric(text),
    logical = as.logical(toupper(text)),
    datetime = as.POSIXct(sub("Z$", "", text), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC"),
    date = as.Date(text)
  )
}

#' Compare two snapshots of a listing
#'
#' @param old,new Data frames from [records.to.frame()] with a key column.
#' @param key Name of the key column.
#' @return A list with elements `added` (rows of `new` with a new key),
#'   `removed` (rows of `old` whose key is gone) and `changed` (rows of
#'   `new` whose values differ from `old`).
#' @export
records.diff <- function(old, new, key = "id") {
  if (!key %in% names(old) || !key %in% names(new)) {
    stop(sprintf("Both snapshots need the key column `%s`.", key), call. = FALSE)
  }

  # Added: keys only in `new`
  added <- new[!new[[key]] %in% old[[key]], , drop = FALSE]
  rownames(added) <- NULL
  # Removed: keys only in `old`
  removed <- old[!old[[key]] %in% new[[key]], , drop = FALSE]
  rownames(removed) <- NULL
  # Changed: shared keys whose shared columns differ
  shared <- intersect(new[[key]], old[[key]])
  columns <- setdiff(intersect(names(old), names(new)), key)
  is.changed <- vapply(shared, function(k) {
    a <- old[old[[key]] == k, columns, drop = FALSE]
    b <- new[new[[key]] == k, columns, drop = FALSE]
    !identical(unname(as.list(a)), unname(as.list(b)))
  }, logical(1))
  changed <- new[new[[key]] %in% shared[is.changed], , drop = FALSE]
  rownames(changed) <- NULL
  list(added = added, removed = removed, changed = changed)
}

#' Extract a value from a nested list by path
#'
#' Walks `x` following the elements of `path`, which may be names or integer
#' positions, and returns `default` as soon as a step is missing.
#'
#' @param x A list, usually parsed from JSON.
#' @param path Character or list path, e.g. `c("meta", "page")`.
#' @param default Value returned when the path does not exist.
#' @return The value found at `path`, or `default`.
#' @export
pluckPath = function(x, path, default = NULL) {
  for (step in path) {
    if (is.null(x)) {
      return(default)
    }
    if (is.numeric(step)) {
      if (step > length(x)) return(default)
      x = x[[step]]
    } else {
      if (!step %in% names(x)) return(default)
      x = x[[step]]
    }
  }
  return(if (is.null(x)) default else x)
}

#' Extract a character scalar from a nested list
#'
#' @inheritParams pluckPath
#' @return A length-one character vector.
#' @export
pluckChr = function(x, path, default = NA_character_) {
  value = pluckPath(x, path, default)
  if (length(value) != 1L) {
    return(default)
  }
  return(as.character(value))
}

#' Extract an integer scalar from a nested list
#'
#' @inheritParams pluckPath
#' @return A length-one integer vector.
#' @export
pluckInt = function(x, path, default = NA_integer_) {
  value = pluckPath(x, path, default)
  if (length(value) != 1L) {
    return(default)
  }
  return(suppressWarnings(as.integer(value)))
}

#' Extract a logical scalar from a nested list
#'
#' @inheritParams pluckPath
#' @return A length-one logical vector.
#' @export
pluckLgl = function(x, path, default = NA) {
  value = pluckPath(x, path, default)
  if (length(value) != 1L) {
    return(default)
  }
  return(as.logical(value))
}

#' Flatten a nested list into a single-row data frame
#'
#' Names of nested elements are joined with `sep`. Unnamed vectors of length
#' greater than one are collapsed into a single string.
#'
#' @param x A list, usually parsed from JSON.
#' @param sep Separator used when joining nested names.
#' @param prefix Name prefix used during recursion.
#' @return A one-row data frame.
#' @export
flattenRecord = function(x, sep = ".", prefix = NULL) {
  out = list()
  for (nm in names(x)) {
    key = if (is.null(prefix)) nm else paste(prefix, nm, sep = sep)
    value = x[[nm]]
    if (is.list(value) && !is.null(names(value))) {
      nested = flattenRecord(value, sep = sep, prefix = key)
      out = c(out, as.list(nested))
    } else if (length(value) == 0L) {
      out[[key]] = NA
    } else if (length(value) > 1L || is.list(value)) {
      out[[key]] = paste(unlist(value), collapse = "|")
    } else {
      out[[key]] = value
    }
  }
  return(as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE))
}

#' Parse JSON log lines into a data frame
#'
#' Each line is parsed separately so that one malformed line does not stop the
#' whole file; failures are collected in the `"errors"` attribute.
#'
#' @param lines Character vector of JSON lines.
#' @param sep Separator used when joining nested names.
#' @return A data frame with one row per successfully parsed line.
#' @export
parseJsonLines = function(lines, sep = ".") {
  lines = lines[nzchar(trimws(lines))]
  rows = vector("list", length(lines))
  errors = character(0)
  for (i in seq_along(lines)) {
    rows[[i]] = tryCatch(
      flattenRecord(jsonlite::fromJSON(lines[[i]], simplifyVector = FALSE), sep = sep),
      error = function(e) {
        errors <<- c(errors, sprintf("line %d: %s", i, conditionMessage(e)))
        NULL
      }
    )
  }
  rows = Filter(Negate(is.null), rows)
  out = if (length(rows) > 0L) bindRowsFill(rows) else data.frame()
  attr(out, "errors") = errors
  return(out)
}

#' Bind data frames by column name, filling missing columns
#'
#' @param frames A list of data frames.
#' @return A single data frame containing the union of all columns.
#' @keywords internal
bindRowsFill = function(frames) {
  allCols = unique(unlist(lapply(frames, names)))
  filled = lapply(frames, function(df) {
    missing = setdiff(allCols, names(df))
    for (col in missing) {
      df[[col]] = NA
    }
    df[allCols]
  })
  out = do.call(rbind, filled)
  rownames(out) = NULL
  return(out)
}

#' Extract typed fields from JSON events according to a schema
#'
#' The schema maps output column names to a JSON path and a type. Each event
#' contributes one row; fields that are missing or of the wrong shape become
#' `NA`, and the number of such misses per column is stored in `"misses"`.
#'
#' @param events A list of parsed JSON objects.
#' @param schema A named list; each element is a list with `path` and `type`,
#'   where `type` is one of `"chr"`, `"int"`, `"lgl"` or `"time"`.
#' @return A data frame with one column per schema entry.
#' @export
extractEvents = function(events, schema) {
  validTypes = c("chr", "int", "lgl", "time")
  types = vapply(schema, function(s) s$type, character(1))
  bad = setdiff(types, validTypes)
  if (length(bad) > 0L) {
    stop("Unknown schema type(s): ", paste(bad, collapse = ", "), call. = FALSE)
  }
  n = length(events)
  out = vector("list", length(schema))
  names(out) = names(schema)
  misses = integer(length(schema))
  names(misses) = names(schema)
  for (col in names(schema)) {
    spec = schema[[col]]
    values = switch(spec$type,
      chr = vapply(events, pluckChr, character(1), path = spec$path),
      int = vapply(events, pluckInt, integer(1), path = spec$path),
      lgl = vapply(events, pluckLgl, logical(1), path = spec$path),
      time = vapply(events, pluckChr, character(1), path = spec$path)
    )
    if (spec$type == "time") {
      values = as.POSIXct(values, format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC")
    }
    misses[[col]] = sum(is.na(values))
    out[[col]] = values
  }
  out = as.data.frame(out, stringsAsFactors = FALSE)
  if (n == 0L) {
    out = out[0, , drop = FALSE]
  }
  attr(out, "misses") = misses
  return(out)
}

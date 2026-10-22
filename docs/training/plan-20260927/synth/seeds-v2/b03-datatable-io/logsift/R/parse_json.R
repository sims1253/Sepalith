#' Extract a value from a nested list by path
#'
#' @description Walks `x` following the elements of `path`, which may be names
#' or integer positions, and returns `default` as soon as a step is missing.
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
    # Positions and names need different existence checks
    if (is.numeric(step)) {
      if (step < 1 || step > length(x)) return(default)
      x = x[[step]]
    } else {
      if (!step %in% names(x)) return(default)
      x = x[[step]]
    }
  }
  return(if (is.null(x)) default else x)
}

#' Extract a typed scalar from a nested list
#'
#' @description Wrappers around [pluckPath()] that return `default` unless the
#' value found is a single element, and coerce it to the requested type.
#'
#' @inheritParams pluckPath
#' @return A length-one vector of the requested type.
#' @export
pluckChr = function(x, path, default = NA_character_) {
  value = pluckPath(x, path, default)
  if (length(value) != 1L) {
    return(default)
  }
  return(as.character(value))
}

#' @rdname pluckChr
#' @export
pluckInt = function(x, path, default = NA_integer_) {
  value = pluckPath(x, path, default)
  if (length(value) != 1L) {
    return(default)
  }
  return(suppressWarnings(as.integer(value)))
}

#' @rdname pluckChr
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
#' @description Names of nested elements are joined with `sep`. Unnamed
#' vectors of length greater than one are collapsed into a single string.
#'
#' @param x A list, usually parsed from JSON.
#' @param sep Separator used when joining nested names.
#' @param prefix Name prefix used during recursion.
#' @keywords internal
flattenRecord = function(x, sep = ".", prefix = NULL) {
  out = list()

  for (nm in names(x)) {
    key = if (is.null(prefix)) nm else paste(prefix, nm, sep = sep)
    value = x[[nm]]
    if (is.list(value) && !is.null(names(value))) {
      # Named sub-objects recurse and contribute prefixed columns
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
#' @description Each line is parsed separately so that one malformed line does
#' not stop the whole file; failures are collected in the `"errors"` attribute
#' and refer to positions in the original `lines`.
#'
#' @param lines Character vector of JSON lines.
#' @param sep Separator used when joining nested names.
#' @return A data frame with one row per successfully parsed line.
#' @export
parseJsonLines = function(lines, sep = ".") {
  # Remember original positions so error messages point at the right line
  lineNo = which(!is.na(lines) & nzchar(trimws(lines)))
  rows = vector("list", length(lineNo))
  errors = character(0)

  for (i in seq_along(lineNo)) {
    rows[[i]] = tryCatch(
      flattenRecord(jsonlite::fromJSON(lines[[lineNo[i]]], simplifyVector = FALSE), sep = sep),
      error = function(e) {
        errors <<- c(errors, sprintf("line %d: %s", lineNo[i], conditionMessage(e)))
        NULL
      }
    )
  }

  # failed lines left NULL slots behind
  rows = Filter(Negate(is.null), rows)
  out = if (length(rows) > 0L) bindRowsFill(rows) else data.frame()
  attr(out, "errors") = errors
  return(out)
}

#' Bind data frames by column name, filling missing columns
#'
#' @param frames A list of data frames.
#' @keywords internal
bindRowsFill = function(frames) {
  # column order follows first appearance across frames
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
#' @description The schema maps output column names to a JSON path and a type.
#' Each event contributes one row; fields that are missing or of the wrong
#' shape become `NA`, and the number of such misses per column is stored in
#' `"misses"`.
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
    # Timestamps arrive as strings and are parsed after extraction
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

  out = as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
  if (n == 0L) {
    out = out[0, , drop = FALSE]
  }
  attr(out, "misses") = misses
  return(out)
}

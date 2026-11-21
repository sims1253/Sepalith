#' @title Escape strings for JSON
#' @description Escapes backslash, double quote, newline, carriage return and
#'   tab, in that order.
#' @param x A character vector.
#' @return A character vector of escaped strings, without surrounding quotes.
#' @keywords internal
jsonEscape = function(x) {
  out = gsub("\\", "\\\\", x, fixed = TRUE)
  out = gsub("\"", "\\\"", out, fixed = TRUE)
  out = gsub("\n", "\\n", out, fixed = TRUE)
  out = gsub("\r", "\\r", out, fixed = TRUE)
  out = gsub("\t", "\\t", out, fixed = TRUE)
  out
}

#' @title Unescape JSON strings
#' @description Reverses [jsonEscape()]. Escaped backslashes are parked on a
#'   placeholder byte first so that `\\n` is not read as a newline.
#' @param x A character vector of escaped strings, without surrounding quotes.
#' @return A character vector.
#' @keywords internal
jsonUnescape = function(x) {
  out = gsub("\\\\", "\001", x, fixed = TRUE)
  out = gsub("\\\"", "\"", out, fixed = TRUE)
  out = gsub("\\n", "\n", out, fixed = TRUE)
  out = gsub("\\r", "\r", out, fixed = TRUE)
  out = gsub("\\t", "\t", out, fixed = TRUE)
  out = gsub("\001", "\\", out, fixed = TRUE)
  out
}

#' @title Encode an atomic vector as JSON
#' @description Length-one vectors become JSON scalars, longer vectors become
#'   arrays. `NA` becomes `null`.
#' @param x A character, factor, numeric, integer, logical, Date or POSIXct
#'   vector. Factors are written as their labels and date-times as UTC ISO
#'   8601 strings.
#' @return A single string.
#' @export
jsonValue = function(x) {
  kind = class(x)[1]
  if (length(x) == 0) {
    return("[]")
  }
  out = switch(kind,
    character = {
      paste0("\"", jsonEscape(x), "\"")
    },
    factor = {
      paste0("\"", jsonEscape(as.character(x)), "\"")
    },
    numeric = {
      as.character(signif(x, 15))
    },
    integer = {
      as.character(x)
    },
    logical = {
      tolower(as.character(x))
    },
    Date = {
      paste0("\"", format(x, "%Y-%m-%d"), "\"")
    },
    POSIXct = {
      paste0("\"", format(x, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), "\"")
    },
    stop(sprintf("Cannot encode an object of class `%s` as JSON.", kind), call. = FALSE)
  )

  out[is.na(x)] = "null"
  if (length(x) == 1) out else paste0("[", paste(out, collapse = ","), "]")
}

#' @title Decode a JSON scalar token
#' @description Mirrors [jsonValue()] for scalars: `null`, `true`, `false`,
#'   quoted strings and numbers.
#' @param token A single JSON token as a string.
#' @return A length-one vector: `NA`, logical, character or numeric.
#' @export
parseJsonValue = function(token) {
  token = trimws(token)
  kind = if (token == "null") {
    "null"
  } else if (token %in% c("true", "false")) {
    "boolean"
  } else if (grepl("^\".*\"$", token)) {
    "string"
  } else if (grepl("^-?[0-9.]+([eE][-+]?[0-9]+)?$", token)) {
    "number"
  } else {
    stop(sprintf("Cannot decode JSON token `%s`.", token), call. = FALSE)
  }

  switch(kind,
    null = {
      NA
    },
    boolean = {
      token == "true"
    },
    string = {
      jsonUnescape(substr(token, 2, nchar(token) - 1))
    },
    number = {
      as.numeric(token)
    }
  )
}

#' @title Encode a record as a JSON object
#' @description Elements that are themselves named lists become nested
#'   objects; `NULL` elements become `null`.
#' @param record A named list of atomic vectors, named lists or `NULL`.
#' @return A single string holding one JSON object.
#' @export
toJsonObject = function(record) {
  keys = names(record)
  if (length(record) == 0) {
    return("{}")
  }
  if (is.null(keys) || any(keys == "")) {
    stop("`record` must be a fully named list.", call. = FALSE)
  }
  pairs = character(length(record))
  for (i in seq_along(record)) {
    value = record[[i]]
    encoded = if (is.null(value)) {
      "null"
    } else if (is.list(value)) {
      toJsonObject(value)
    } else {
      jsonValue(value)
    }
    pairs[i] = paste0("\"", jsonEscape(keys[i]), "\":", encoded)
  }
  paste0("{", paste(pairs, collapse = ","), "}")
}

#' @title Write a data frame as JSON lines
#' @param data A data frame of atomic columns.
#' @param path Destination file path.
#' @param append Whether to add the lines to an existing file.
#' @return The number of lines written, invisibly.
#' @export
writeJsonLines = function(data, path, append = FALSE) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  lines = character(nrow(data))
  for (i in seq_len(nrow(data))) {
    record = lapply(data, `[`, i)
    lines[i] = toJsonObject(record)
  }

  con = file(path, open = if (append) "a" else "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(lines, con)
  invisible(length(lines))
}

#' @title Read JSON lines of flat objects
#' @description Each line must hold one object whose values are scalars.
#'   Keys missing on some lines are filled with `NA`.
#' @param path Path to a file written by [writeJsonLines()].
#' @return A data frame with one column per key, in order of first
#'   appearance.
#' @export
readJsonLines = function(path) {
  con = file(path, open = "r", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  lines = readLines(con)
  lines = lines[nzchar(trimws(lines))]

  pairPattern = "\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*(\"(?:[^\"\\\\]|\\\\.)*\"|[^,}]+)"
  records = vector("list", length(lines))
  for (i in seq_along(lines)) {
    hits = regmatches(lines[i], gregexpr(pairPattern, lines[i], perl = TRUE))[[1]]
    keys = jsonUnescape(sub(pairPattern, "\\1", hits, perl = TRUE))
    tokens = sub(pairPattern, "\\2", hits, perl = TRUE)
    records[[i]] = stats::setNames(lapply(tokens, parseJsonValue), keys)
  }

  keys = unique(unlist(lapply(records, names)))
  columns = lapply(keys, function(key) {
    unlist(lapply(records, function(rec) if (is.null(rec[[key]])) NA else rec[[key]]))
  })
  data = as.data.frame(stats::setNames(columns, keys), stringsAsFactors = FALSE)
  data
}

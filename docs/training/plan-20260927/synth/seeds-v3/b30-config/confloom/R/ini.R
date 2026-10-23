#' Classify one line of an INI-like file
#'
#' @param line A single line of text, untrimmed.
#' @return One of `"blank"`, `"comment"`, `"section"`, `"pair"`,
#'   `"continuation"` or `"invalid"`. Indented lines that are not blank or
#'   comments continue the previous value.
#' @noRd
classifyIniLine = function(line) {
  trimmed = trimws(line)
  if (!nzchar(trimmed)) {
    "blank"
  } else if (substr(trimmed, 1, 1) %in% c("#", ";")) {
    "comment"
  } else if (grepl("^[[:space:]]", line)) {
    "continuation"
  } else if (grepl("^\\[[^]]+\\]$", trimmed)) {
    "section"
  } else if (grepl("^[^=]+=", trimmed)) {
    "pair"
  } else {
    "invalid"
  }
}

#' Read an INI-like configuration file
#'
#' @description
#' Keys before the first section header go into the section named by
#' `global`.
#'
#' @param path Path to the file.
#' @param global Name of the section for keys before the first header.
#' @param strict If `TRUE`, malformed lines are an error; otherwise they are
#'   skipped with a warning.
#' @details Indented lines continue the value of the previous key, joined
#'   with a newline. Later duplicates of a key overwrite earlier ones.
#'   Values are kept as strings.
#' @return A named list of sections, each a named list of string values.
#' @export
readIniConfig = function(path, global = "global", strict = TRUE) {
  lines = readLines(path, warn = FALSE)
  out = list()
  section = global
  key = NULL

  for (i in seq_along(lines)) {
    line = lines[[i]]
    kind = classifyIniLine(line)
    # A continuation without a preceding key is malformed
    if (kind == "continuation" && is.null(key)) kind = "invalid"

    switch(kind,
      blank = ,
      comment = {
        next
      },
      section = {
        line = trimws(line)
        section = trimws(substr(line, 2, nchar(line) - 1))
        key = NULL
        if (is.null(out[[section]])) out[[section]] = list()
      },
      pair = {
        line = trimws(line)
        eq = regexpr("=", line, fixed = TRUE)
        key = trimws(substr(line, 1, eq - 1))
        value = trimws(substr(line, eq + 1, nchar(line)))
        if (is.null(out[[section]])) out[[section]] = list()
        out[[section]][[key]] = value
      },
      continuation = {
        previous = out[[section]][[key]]
        out[[section]][[key]] = paste(previous, trimws(line), sep = "\n")
      },
      invalid = {
        msg = sprintf("Line %d of '%s' is not a section, pair or comment.", i, path)
        if (strict) stop(msg, call. = FALSE)
        warning(msg, call. = FALSE)
      }
    )
  }
  out
}

#' Write an INI-like configuration file
#'
#' @description
#' The inverse of [readIniConfig()]. The section named by `global` is
#' written first without a header; every other section gets a `[name]`
#' header followed by its `key = value` lines. Values containing newlines
#' are written as indented continuation lines.
#'
#' @param config A named list of sections, each a named list of values.
#' @param path Path to the file.
#' @param global Name of the section for keys before the first header.
#' @param header Optional comment lines written at the top of the file.
#' @export
writeIniConfig = function(config, path, global = "global", header = NULL) {
  formatPairs = function(values) {
    values = vapply(values, function(v) paste(format(v), collapse = ", "),
                    character(1))
    values = gsub("\n", "\n    ", values, fixed = TRUE)
    paste(names(values), "=", values)
  }

  lines = character()
  if (length(header) > 0) {
    lines = c(lines, paste("#", header))
  }
  if (length(config[[global]]) > 0) {
    lines = c(lines, formatPairs(config[[global]]))
  }
  for (section in setdiff(names(config), global)) {
    if (length(lines) > 0) lines = c(lines, "")
    lines = c(lines, sprintf("[%s]", section))
    if (length(config[[section]]) > 0) {
      lines = c(lines, formatPairs(config[[section]]))
    }
  }

  writeLines(lines, path)
  invisible(path)
}

#' Get a typed INI value
#'
#' @param config A named list of sections, each a named list of values.
#' @param section Section name.
#' @param key Key within the section.
#' @param default Value returned when the key is absent.
#' @param type How to convert the stored string.
#' @return The converted value or `default`. Errors when the value cannot be
#'   converted.
#' @export
iniGet = function(config, section, key, default = NULL,
                  type = c("string", "integer", "number", "boolean", "list")) {
  type = match.arg(type)
  values = config[[section]]
  if (is.null(values) || !key %in% names(values)) {
    return(default)
  }

  # Convert the stored string; failed conversions become NA
  raw = values[[key]]
  out = switch(type,
    string = as.character(raw),
    integer = suppressWarnings(as.integer(raw)),
    number = suppressWarnings(as.numeric(raw)),
    boolean = switch(tolower(raw), true = , yes = , on = , `1` = TRUE,
                     false = , no = , off = , `0` = FALSE, NA),
    list = trimws(strsplit(raw, "[,\n]")[[1]])
  )
  if (anyNA(out)) {
    stop(sprintf("Value of [%s] %s is not a valid %s: \"%s\".", section, key,
                 type, raw), call. = FALSE)
  }
  out
}

#' Set an INI value
#'
#' @inheritParams iniGet
#' @param value New value for the key, or `NULL` to remove the key.
#' @keywords internal
iniSet = function(config, section, key, value) {
  if (is.null(config[[section]])) {
    config[[section]] = list()
  }
  config[[section]][[key]] = value
  config
}

#' Check INI values against a schema
#'
#' @description
#' The schema names, per section, the type of every key. Each value is
#' tested with the same conversion rules as [iniGet()]; keys missing from
#' the file are reported when `required` is `TRUE`.
#'
#' @param config A named list of sections, each a named list of values.
#' @param schema A named list of sections, each a named character vector
#'   of types: `"string"`, `"integer"`, `"number"`, `"boolean"` or
#'   `"list"`.
#' @param required Whether every key in `schema` must be present.
#' @return A character vector of problems, empty when `config` matches.
#' @export
checkIniTypes = function(config, schema, required = FALSE) {
  problems = character()
  for (section in names(schema)) {
    for (key in names(schema[[section]])) {
      type = schema[[section]][[key]]
      raw = config[[section]][[key]]
      if (is.null(raw)) {
        if (required) {
          problems = c(problems, sprintf("[%s] %s is missing.", section, key))
        }
        next
      }

      ok = switch(type,
        string = {
          is.character(raw) && length(raw) == 1
        },
        integer = {
          grepl("^-?[0-9]+$", raw)
        },
        number = {
          !is.na(suppressWarnings(as.numeric(raw)))
        },
        boolean = {
          tolower(raw) %in% c("true", "yes", "on", "1", "false", "no", "off", "0")
        },
        list = {
          all(nzchar(trimws(strsplit(raw, "[,\n]")[[1]])))
        },
        stop(sprintf("Unknown type '%s' in schema for [%s] %s.", type, section, key),
             call. = FALSE)
      )

      if (!ok) {
        problems = c(problems, sprintf("[%s] %s is not a valid %s.", section, key, type))
      }
    }
  }
  problems
}

#' Expand references between INI values
#'
#' @description
#' A value may refer to another value as `${key}` (same section) or
#' `${section:key}`.
#'
#' @param config A named list of sections, each a named list of string
#'   values.
#' @param maxDepth Maximum number of expansion rounds.
#' @details References are expanded repeatedly until none remain, up
#'   to `maxDepth` rounds, which also guards against cycles.
#' @return `config` with references replaced.
#' @keywords internal
interpolateIni = function(config, maxDepth = 10) {
  pattern = "\\$\\{([A-Za-z0-9_.-]+:)?[A-Za-z0-9_.-]+\\}"
  # Each round replaces every reference found in the current values
  for (round in seq_len(maxDepth)) {
    changed = FALSE
    for (section in names(config)) {
      for (key in names(config[[section]])) {
        value = config[[section]][[key]]
        refs = regmatches(value, gregexpr(pattern, value))[[1]]
        for (ref in unique(refs)) {
          target = strsplit(substr(ref, 3, nchar(ref) - 1), ":", fixed = TRUE)[[1]]
          if (length(target) == 1) target = c(section, target)
          replacement = iniGet(config, target[[1]], target[[2]])
          if (is.null(replacement)) {
            stop(sprintf("Reference %s in [%s] %s is undefined.", ref, section, key),
                 call. = FALSE)
          }
          value = gsub(ref, replacement, value, fixed = TRUE)
          changed = TRUE
        }
        config[[section]][[key]] = value
      }
    }
    if (!changed) return(config)
  }

  stop(sprintf("References still unresolved after %d rounds.", maxDepth),
       call. = FALSE)
}

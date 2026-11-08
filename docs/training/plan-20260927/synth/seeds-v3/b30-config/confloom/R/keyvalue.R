#' Quote or unquote values for key=value files
#'
#' @description
#' When encoding, values containing spaces, quotes, `#` or the separator
#' are wrapped in double quotes with embedded quotes and backslashes
#' escaped. Decoding reverses the transformation.
#'
#' @param x A character vector of values.
#' @param direction Whether to `"encode"` or `"decode"`.
#' @param sep The key/value separator.
#' @keywords internal
escapeValue = function(x, direction = c("encode", "decode"), sep = "=") {
  direction = match.arg(direction)
  if (direction == "encode") {
    hit = grepl("[[:space:]\"#\\\\]", x) | grepl(sep, x, fixed = TRUE) | !nzchar(x)
    inner = x[hit]
    inner = gsub("\\", "\\\\", inner, fixed = TRUE)
    inner = gsub("\"", "\\\"", inner, fixed = TRUE)
    x[hit] = paste0("\"", inner, "\"")
  } else {
    hit = grepl("^\".*\"$", x) & nchar(x) >= 2
    inner = substr(x[hit], 2, nchar(x[hit]) - 1)
    inner = gsub("\\\"", "\"", inner, fixed = TRUE)
    inner = gsub("\\\\", "\\", inner, fixed = TRUE)
    x[hit] = inner
  }
  x
}

#' Parse one line of a key=value file
#'
#' @param line A single line of text.
#' @param sep The key/value separator.
#' @param comment The comment character.
#' @return A length-one named character vector, or `NULL` for blank and
#'   comment lines.
#' @noRd
parseKeyValueLine = function(line, sep = "=", comment = "#") {
  line = trimws(line)
  if (!nzchar(line) || startsWith(line, comment)) {
    return(NULL)
  }

  # Allow shell-style `export KEY=value`
  line = sub("^export[[:space:]]+", "", line)
  pos = regexpr(sep, line, fixed = TRUE)
  if (pos < 2) {
    stop(sprintf("Cannot parse line \"%s\".", line), call. = FALSE)
  }
  key = trimws(substr(line, 1, pos - 1))
  value = trimws(substr(line, pos + nchar(sep), nchar(line)))
  if (!startsWith(value, "\"")) {
    value = trimws(sub(paste0("[[:space:]]+", comment, ".*$"), "", value))
  }
  stats::setNames(escapeValue(value, "decode", sep), key)
}

#' Read a key=value configuration file
#'
#' @description
#' Values may refer to keys defined earlier in the file, or to environment
#' variables, as `${NAME}`. Keys from the file take precedence.
#'
#' @param path Path to the file.
#' @param sep The key/value separator.
#' @param comment The comment character.
#' @param expand Whether to expand `${NAME}` references.
#' @return A named character vector; later duplicates of a key win with a
#'   warning.
#' @export
readKvConfig = function(path, sep = "=", comment = "#", expand = TRUE) {
  lines = readLines(path, warn = FALSE)
  out = character()
  for (line in lines) {
    pair = parseKeyValueLine(line, sep, comment)
    if (is.null(pair)) next
    key = names(pair)
    if (key %in% names(out)) {
      warning(sprintf("Key '%s' is set more than once in '%s'.", key, path),
              call. = FALSE)
    }
    value = pair[[1]]

    # Expand references to earlier keys, then to environment variables
    if (expand) {
      refs = unique(regmatches(value, gregexpr("\\$\\{[A-Za-z_][A-Za-z0-9_]*\\}", value))[[1]])
      for (ref in refs) {
        name = substr(ref, 3, nchar(ref) - 1)
        known = if (name %in% names(out)) out[[name]] else Sys.getenv(name)
        value = gsub(ref, known, value, fixed = TRUE)
      }
    }
    out[key] = value
  }
  out
}

#' Write a key=value configuration file
#'
#' @description
#' Keys are written in sorted order. With `group = TRUE`, a blank line
#' separates keys with different prefixes (the part before the first `_`).
#'
#' @param config A named character vector or named list of scalars.
#' @param path Path to the file.
#' @param sep The key/value separator.
#' @param comment The comment character.
#' @param header Optional comment lines written at the top of the file.
#' @param group Whether to separate prefix groups by blank lines.
#' @export
writeKvConfig = function(config, path, sep = "=", comment = "#", header = NULL,
                         group = FALSE) {
  values = vapply(config, function(v) format(v), character(1))
  if (is.null(names(values)) || any(!nzchar(names(values)))) {
    stop("Every value in `config` must be named.", call. = FALSE)
  }

  values = values[order(names(values))]
  lines = paste0(names(values), sep, escapeValue(values, "encode", sep))
  if (group && length(lines) > 1) {
    prefix = sub("_.*$", "", names(values))
    starts = which(prefix[-1] != prefix[-length(prefix)]) + 1
    for (at in rev(starts)) {
      lines = append(lines, "", after = at - 1)
    }
  }
  if (length(header) > 0) {
    lines = c(paste(comment, header), "", lines)
  }

  writeLines(lines, path)
  invisible(path)
}

#' Read a configuration file in any supported format
#'
#' @description
#' Every reader's result is normalised to a named list and tagged with an
#' attribute `format`.
#'
#' @param path Path to the file.
#' @param format File format; guessed from the extension when missing.
#' @return A named list.
#' @export
readConfig = function(path, format = c("dcf", "ini", "kv", "json", "yaml")) {
  if (!file.exists(path)) {
    stop(sprintf("Config file '%s' does not exist.", path), call. = FALSE)
  }

  # Guess the format from the file extension
  if (missing(format)) {
    ext = tolower(tools::file_ext(path))
    format = switch(ext, env = , properties = "kv", yml = "yaml", cfg = "ini", ext)
  }
  format = match.arg(format)

  out = switch(format,
    dcf = {
      readDcfConfig(path)
    },
    ini = {
      readIniConfig(path)
    },
    kv = {
      as.list(readKvConfig(path))
    },
    json = {
      jsonlite::read_json(path, simplifyVector = FALSE)
    },
    yaml = {
      yaml::read_yaml(path)
    }
  )
  if (!is.list(out) || (length(out) > 0 && is.null(names(out)))) {
    stop(sprintf("Config file '%s' does not contain named settings.", path),
         call. = FALSE)
  }
  attr(out, "format") = format
  out
}

#' Write a configuration file in any supported format
#'
#' @param config A named list, e.g. as returned by [readConfig()].
#' @param path Path to the file.
#' @param format File format; taken from the `format` attribute of `config`
#'   or guessed from the extension when missing.
#' @export
writeConfig = function(config, path, format = c("dcf", "ini", "kv", "json", "yaml")) {
  # Guess the format from the file extension
  if (missing(format)) {
    ext = tolower(tools::file_ext(path))
    guess = switch(ext, env = , properties = "kv", yml = "yaml", cfg = "ini", ext)
    format = if (is.null(attr(config, "format"))) guess else attr(config, "format")
  }
  format = match.arg(format)
  attr(config, "format") = NULL

  switch(format,
    dcf = {
      writeDcfConfig(config, path)
    },
    ini = {
      writeIniConfig(config, path)
    },
    kv = {
      writeKvConfig(unlist(config), path)
    },
    json = {
      jsonlite::write_json(config, path, auto_unbox = TRUE, pretty = TRUE)
    },
    yaml = {
      yaml::write_yaml(config, path)
    }
  )
  invisible(path)
}

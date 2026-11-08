#' Split identifiers into lower-case words
#'
#' @description
#' Word boundaries are underscores, hyphens, dots, white space and changes
#' from lower to upper case. Runs of capitals such as `"HTTP"` stay together.
#'
#' @param x A character vector of identifiers.
#' @return A list with one character vector of words per element of `x`;
#'   missing identifiers give an empty vector.
#' @export
splitIdentifier = function(x) {
  x[is.na(x)] = ""
  x = gsub("([[:lower:][:digit:]])([[:upper:]])", "\\1 \\2", x)
  x = gsub("([[:upper:]]+)([[:upper:]][[:lower:]])", "\\1 \\2", x)
  x = gsub("[-_.[:space:]]+", " ", x)
  words = strsplit(tolower(trimws(x)), " ", fixed = TRUE)
  lapply(words, function(w) w[w != ""])
}

#' Convert identifiers to a named case
#'
#' @param x A character vector of identifiers.
#' @param to The target case. `"constant"` is upper-case snake case.
#' @param prefix Optional word prepended to every identifier before
#'   conversion.
#' @return A character vector the same length as `x`; `NA` stays `NA`.
#' @family case converters
#' @export
convertCase = function(x, to = c("snake", "kebab", "dot", "camel", "pascal", "constant"),
                       prefix = NULL) {
  to = match.arg(to)
  missingId = is.na(x)
  if (!is.null(prefix)) {
    x = paste(prefix, x)
  }

  # Split into words, join them per case, then restore missing values
  words = splitIdentifier(x)
  out = switch(to,
    snake = {
      vapply(words, function(w) {
        paste(w, collapse = "_")
      }, character(1))
    },
    kebab = {
      vapply(words, function(w) {
        paste(w, collapse = "-")
      }, character(1))
    },
    dot = {
      vapply(words, function(w) {
        paste(w, collapse = ".")
      }, character(1))
    },
    camel = {
      vapply(words, function(w) {
        upper = paste0(toupper(substr(w, 1L, 1L)), substr(w, 2L, nchar(w)))
        paste(c(utils::head(w, 1L), upper[-1]), collapse = "")
      }, character(1))
    },
    pascal = {
      vapply(words, function(w) {
        upper = paste0(toupper(substr(w, 1L, 1L)), substr(w, 2L, nchar(w)))
        paste(upper, collapse = "")
      }, character(1))
    },
    constant = {
      vapply(words, function(w) {
        paste(toupper(w), collapse = "_")
      }, character(1))
    }
  )

  out[missingId] = NA_character_
  unname(out)
}

#' Guess the case of identifiers
#'
#' @param x A character vector of identifiers.
#' @return A character vector with one of `"snake"`, `"kebab"`, `"dot"`,
#'   `"camel"`, `"pascal"`, `"constant"` or `"other"` per element of `x`.
#'   Single lower-case words count as `"snake"`.
#' @export
detectCase = function(x) {
  patterns = c(
    snake = "^[a-z][a-z0-9]*(_[a-z0-9]+)*$",
    kebab = "^[a-z][a-z0-9]*(-[a-z0-9]+)+$",
    dot = "^[a-z][a-z0-9]*([.][a-z0-9]+)+$",
    camel = "^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$",
    pascal = "^[A-Z][a-z0-9]+([A-Z][a-z0-9]*)*$",
    constant = "^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"
  )

  out = rep("other", length(x))
  for (case in names(patterns)) {
    hit = out == "other" & grepl(patterns[[case]], x)
    out[hit] = case
  }
  out[is.na(x)] = NA_character_
  out
}

#' Convert text to title case
#'
#' @description
#' Capitalises every word except the minor words in `minor`. The first and
#' the last word are always capitalised.
#'
#' @param x A character vector of titles.
#' @param minor Lower-case words kept in lower case inside a title.
#' @return A character vector the same length as `x`.
#' @export
titleCase = function(x, minor = c("a", "an", "and", "as", "at", "but", "by", "for", "in",
                                  "of", "on", "or", "the", "to", "with")) {
  out = x
  for (i in seq_along(x)) {
    if (is.na(x[i])) {
      next
    }
    words = strsplit(tolower(trimws(x[i])), "[[:space:]]+")[[1]]
    n = length(words)
    if (n == 0L) {
      out[i] = ""
      next
    }
    cap = !words %in% minor
    # The first word is always capitalised
    cap[1] = TRUE
    # The last word is always capitalised
    cap[n] = TRUE
    first = toupper(substr(words[cap], 1L, 1L))
    words[cap] = paste0(first, substr(words[cap], 2L, nchar(words[cap])))
    out[i] = paste(words, collapse = " ")
  }
  out
}

#' Profile the letter case of strings
#'
#' @param x A character vector.
#' @return A data frame with one row per element of `x` and columns
#'   `nUpper`, `nLower`, `nDigit`, `nOther` and `shareUpper` (upper-case
#'   letters among all letters).
#' @export
caseProfile = function(x) {
  x[is.na(x)] = ""
  nChar = nchar(x)
  nUpper = nChar - nchar(gsub("[[:upper:]]", "", x))
  nLower = nChar - nchar(gsub("[[:lower:]]", "", x))
  nDigit = nChar - nchar(gsub("[[:digit:]]", "", x))
  nOther = nChar - nUpper - nLower - nDigit

  letterCount = nUpper + nLower
  shareUpper = ifelse(letterCount > 0L, nUpper / letterCount, NA_real_)
  data.frame(
    nUpper = nUpper,
    nLower = nLower,
    nDigit = nDigit,
    nOther = nOther,
    shareUpper = shareUpper
  )
}

#' Rename the columns of a data frame to one case
#'
#' @param data A data frame.
#' @param to The target case, passed to [convertCase()].
#' @return `data` with renamed columns. Duplicated new names are made unique
#'   and the attribute `renamed` maps old names to new ones.
#' @export
renameColumns = function(data, to = "snake") {
  oldNames = names(data)
  newNames = convertCase(oldNames, to = to)
  # Columns whose names have no words keep their old name
  empty = is.na(newNames) | newNames == ""
  newNames[empty] = oldNames[empty]
  if (anyDuplicated(newNames) > 0L) {
    warning("Renaming produced duplicate names; making them unique.", call. = FALSE)
    newNames = make.unique(newNames, sep = "_")
  }
  names(data) = newNames
  changed = oldNames != newNames
  attr(data, "renamed") = stats::setNames(newNames[changed], oldNames[changed])
  data
}

#' Trim and collapse whitespace
#'
#' @param x A character vector.
#' @return `x` with leading and trailing whitespace removed and inner runs
#'   of whitespace replaced by a single space.
#' @export
squish.text <- function(x) {
  gsub("[[:space:]]+", " ", trimws(x))
}

#' Change the letter case of text
#'
#' @param x A character vector.
#' @param case Target case: one of `"lower"`, `"upper"`, `"title"`,
#'   `"sentence"`, `"snake"`, `"kebab"` or `"camel"`.
#' @return A character vector the same length as `x`.
#' @export
change.case <- function(x,
                        case = c("lower", "upper", "title", "sentence", "snake", "kebab",
                                 "camel")) {
  case <- match.arg(case)
  x <- squish.text(x)
  switch(case,
    lower = {
      tolower(x)
    },
    upper = {
      toupper(x)
    },
    title = {
      x <- tolower(x)
      gsub("(^|[[:space:]-])([a-z])", "\\1\\U\\2", x, perl = TRUE)
    },
    sentence = {
      x <- tolower(x)
      sub("^([a-z])", "\\U\\1", x, perl = TRUE)
    },
    snake = {
      x <- gsub("([a-z0-9])([A-Z])", "\\1_\\2", x)
      gsub("[^a-z0-9]+", "_", tolower(x))
    },
    kebab = {
      x <- gsub("([a-z0-9])([A-Z])", "\\1-\\2", x)
      gsub("[^a-z0-9]+", "-", tolower(x))
    },
    camel = {
      x <- gsub("([a-z0-9])([A-Z])", "\\1 \\2", x)
      x <- gsub("[^a-z0-9]+(.)", "\\U\\1", tolower(x), perl = TRUE)
      sub("^[^a-z0-9]+", "", x)
    }
  )
}

#' Strip a prefix and a suffix
#'
#' @param x A character vector.
#' @param prefix,suffix Literal strings to remove from the start and the
#'   end of each element; `NULL` skips that side.
#' @param ignore.case Whether matching ignores case.
#' @return `x` without the prefix and suffix.
#' @export
strip.affix <- function(x, prefix = NULL, suffix = NULL, ignore.case = FALSE) {
  fold <- if (ignore.case) tolower else identity
  if (!is.null(prefix)) {
    n <- nchar(prefix)
    hit <- !is.na(x) & fold(substr(x, 1L, n)) == fold(prefix)
    x[hit] <- substr(x[hit], n + 1L, nchar(x[hit]))
  }
  if (!is.null(suffix)) {
    n <- nchar(suffix)
    hit <- !is.na(x) & fold(substr(x, nchar(x) - n + 1L, nchar(x))) == fold(suffix)
    x[hit] <- substr(x[hit], 1L, nchar(x[hit]) - n)
  }
  x
}

#' Pad text to a fixed width
#'
#' @param x A character vector.
#' @param width Target width in characters.
#' @param side Where to add padding: `"left"`, `"right"` or `"both"`.
#' @param pad A single padding character.
#' @return A character vector; elements already at least `width` wide are
#'   unchanged.
#' @export
pad.text <- function(x, width, side = c("left", "right", "both"), pad = " ") {
  side <- match.arg(side)
  if (!is.character(pad) || length(pad) != 1L || nchar(pad) != 1L) {
    stop("`pad` must be a single character.", call. = FALSE)
  }
  missing <- pmax(width - nchar(x), 0L)
  missing[is.na(x)] <- 0L
  if (side == "left") {
    left <- missing
    right <- integer(length(x))
  } else if (side == "right") {
    left <- integer(length(x))
    right <- missing
  } else {
    left <- missing %/% 2L
    right <- missing - left
  }
  paste0(strrep(pad, left), x, strrep(pad, right))
}

#' Normalise column names
#'
#' @param x A character vector of names.
#' @param case Case passed to [change.case()].
#' @param max.len Longest name kept; longer names are truncated.
#' @return A character vector of unique, syntactic names.
#' @export
normalize.names <- function(x, case = "snake", max.len = 64L) {
  # 1. Transliterate to ASCII and squish whitespace
  x <- iconv(x, to = "ASCII//TRANSLIT", sub = "")
  x <- squish.text(x)

  # 2. Apply the requested case
  x <- change.case(x, case = case)

  # 3. Drop separators at the edges and fill blanks
  x <- gsub("^_+|_+$", "", x)
  x[is.na(x) | !nzchar(x)] <- "column"

  # 4. Make the names syntactic and unique
  x[grepl("^[0-9]", x)] <- paste0("x", x[grepl("^[0-9]", x)])
  x <- substr(x, 1L, max.len)
  make.unique(x, sep = "_")
}

#' Normalise text columns of a data frame
#'
#' @param data A data frame to clean.
#' @param columns Names of the character columns to normalise; defaults to
#'   all character columns.
#' @param case Case passed to [change.case()], or `NULL` to keep the case.
#' @param width,side Padding passed to [pad.text()]; `width = NULL` skips
#'   padding.
#' @return `data` with the selected columns normalised.
#' @export
normalize.text.columns <- function(data, columns = NULL, case = NULL, width = NULL,
                                   side = "left") {
  if (is.null(columns)) {
    columns <- names(data)[vapply(data, is.character, logical(1))]
  }
  for (col in columns) {
    value <- squish.text(data[[col]])
    if (!is.null(case)) {
      value <- change.case(value, case = case)
    }
    if (!is.null(width)) {
      value <- pad.text(value, width = width, side = side)
    }
    data[[col]] <- value
  }
  data
}

#' Normalise postal address columns
#'
#' @description
#' Cleans the address columns `street`, `houseNumber`, `city`, `postcode`
#' and `country` that are present in `data`, each in the way it is usually mistyped.
#'
#' @param data A data frame to clean.
#' @param abbreviations Named character vector of street abbreviations to
#'   expand; names are the abbreviations.
#' @return `data` with the address columns normalised.
#' @export
normalize.address <- function(data, abbreviations = c("str." = "strasse", "st." = "street",
                                                      "rd." = "road", "ave." = "avenue")) {
  # Normalise `street`: squish, title case, expand abbreviations
  if ("street" %in% names(data)) {
    street <- tolower(squish.text(data$street))
    for (abbr in names(abbreviations)) {
      street <- gsub(abbr, abbreviations[[abbr]], street, fixed = TRUE)
    }
    data$street <- change.case(street, case = "title")
  }

  # Normalise `houseNumber`: drop spaces, lower case
  if ("houseNumber" %in% names(data)) {
    houseNumber <- gsub("[[:space:]]+", "", data$houseNumber)
    data$houseNumber <- change.case(houseNumber, case = "lower")
  }

  # Normalise `city`: squish, title case
  if ("city" %in% names(data)) {
    city <- squish.text(data$city)
    data$city <- change.case(city, case = "title")
  }

  # Normalise `postcode`: squish, upper case
  if ("postcode" %in% names(data)) {
    postcode <- squish.text(data$postcode)
    data$postcode <- change.case(postcode, case = "upper")
  }

  # Normalise `country`: squish, upper case, strip dots
  if ("country" %in% names(data)) {
    country <- gsub(".", "", squish.text(data$country), fixed = TRUE)
    data$country <- change.case(country, case = "upper")
  }
  data
}

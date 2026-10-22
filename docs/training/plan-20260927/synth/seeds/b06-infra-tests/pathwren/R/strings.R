#' Squish whitespace in strings
#'
#' Trims leading and trailing whitespace and collapses internal runs of
#' whitespace to a single space.
#'
#' @param x A character vector.
#' @return A character vector the same length as `x`.
#' @export
#' @examples
#' strSquish("  too   many\tspaces ")
strSquish = function(x) {
  checkCharacter(x)
  out = gsub("[[:space:]]+", " ", x)
  out = trimws(out)
  return(out)
}

#' Pad strings to a fixed width
#'
#' `strPadLeft()` adds padding on the left so text is right-aligned;
#' `strPadRight()` adds padding on the right so text is left-aligned. Strings
#' already wider than `width` are returned unchanged.
#'
#' @param x A character vector.
#' @param width The target display width.
#' @param pad A single character used for padding.
#' @return A character vector the same length as `x`.
#' @family padding helpers
#' @export
#' @examples
#' strPadLeft(c("7", "42", "512"), 4, pad = "0")
strPadLeft = function(x, width, pad = " ") {
  checkCharacter(x)
  checkPadChar(pad)
  missingWidth = pmax(width - nchar(x, type = "width"), 0L)
  out = paste0(strrep(pad, missingWidth), x)
  out[is.na(x)] = NA_character_
  return(out)
}

#' @rdname strPadLeft
#' @export
strPadRight = function(x, width, pad = " ") {
  checkCharacter(x)
  checkPadChar(pad)
  missingWidth = pmax(width - nchar(x, type = "width"), 0L)
  out = paste0(x, strrep(pad, missingWidth))
  out[is.na(x)] = NA_character_
  return(out)
}

#' Truncate strings to a maximum width
#'
#' Strings longer than `width` are cut and finished with `ellipsis` so that
#' the result is exactly `width` characters wide.
#'
#' @param x A character vector.
#' @param width The target display width.
#' @param ellipsis The string appended to truncated values.
#' @return A character vector the same length as `x`.
#' @export
strTruncate = function(x, width, ellipsis = "...") {
  checkCharacter(x)
  if (width < nchar(ellipsis)) {
    rlang::abort(
      "`width` must be at least as wide as `ellipsis`.",
      class = c("pathwren_error_width", "pathwren_error")
    )
  }
  tooLong = !is.na(x) & nchar(x) > width
  keep = width - nchar(ellipsis)
  x[tooLong] = paste0(substr(x[tooLong], 1L, keep), ellipsis)
  return(x)
}

#' Convert between snake_case and camelCase
#'
#' `strToSnake()` lower-cases words and joins them with underscores.
#' `strToCamel()` capitalises every word but the first and removes separators.
#' Both treat spaces, dots, hyphens and underscores as word separators.
#'
#' @param x A character vector.
#' @return A character vector the same length as `x`.
#' @family case converters
#' @export
#' @examples
#' strToSnake("monthlyNetTotal")
#' strToCamel("monthly_net_total")
strToSnake = function(x) {
  checkCharacter(x)
  out = gsub("([a-z0-9])([A-Z])", "\\1_\\2", x)
  out = gsub("[[:space:].-]+", "_", out)
  out = gsub("_+", "_", out)
  out = tolower(gsub("^_|_$", "", out))
  return(out)
}

#' @rdname strToSnake
#' @export
strToCamel = function(x) {
  checkCharacter(x)
  words = strsplit(strToSnake(x), "_", fixed = TRUE)
  out = vapply(words, function(w) {
    if (length(w) == 0L) {
      return("")
    }
    rest = w[-1L]
    rest = paste0(toupper(substr(rest, 1L, 1L)), substring(rest, 2L))
    return(paste0(c(w[1L], rest), collapse = ""))
  }, character(1))
  out[is.na(x)] = NA_character_
  return(out)
}

#' Count pattern matches in strings
#'
#' @param x A character vector.
#' @param pattern A regular expression, or a fixed string when `fixed = TRUE`.
#' @param fixed Whether `pattern` is matched literally.
#' @return An integer vector the same length as `x`.
#' @export
strCountMatches = function(x, pattern, fixed = FALSE) {
  checkCharacter(x)
  matches = gregexpr(pattern, x, fixed = fixed)
  counts = vapply(matches, function(m) sum(m > 0L), integer(1))
  counts[is.na(x)] = NA_integer_
  return(counts)
}

#' Check that an argument is a character vector
#'
#' @param x The object to check.
#' @param arg The name of the argument being checked, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @noRd
checkCharacter = function(x, arg = deparse(substitute(x))) {
  if (!is.character(x)) {
    rlang::abort(
      sprintf("`%s` must be a character vector, not %s.", arg, class(x)[1L]),
      class = c("pathwren_error_type", "pathwren_error")
    )
  }
  return(invisible(x))
}

#' Check that an argument is a single padding character
#'
#' @param x The object to check.
#' @param arg The name of the argument being checked, used in error messages.
#' @return `x`, invisibly, if the check passes.
#' @noRd
checkPadChar = function(x, arg = deparse(substitute(x))) {
  if (!is.character(x) || length(x) != 1L || nchar(x) != 1L) {
    rlang::abort(
      sprintf("`%s` must be a single character.", arg),
      class = c("pathwren_error_pad", "pathwren_error")
    )
  }
  return(invisible(x))
}

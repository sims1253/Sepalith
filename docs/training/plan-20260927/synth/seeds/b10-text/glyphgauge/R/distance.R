#' Split a string into single characters
#'
#' @param x A single string.
#' @return A character vector of single characters.
#' @noRd
toChars = function(x) {
  strsplit(x, "", fixed = TRUE)[[1]]
}

#' Check a pair of string arguments
#'
#' @param a,b Single strings to compare.
#' @return `TRUE`, invisibly, or an error.
#' @noRd
checkPair = function(a, b) {
  if (!is.character(a) || length(a) != 1L) {
    stop("`a` must be a single string.", call. = FALSE)
  }
  if (!is.character(b) || length(b) != 1L) {
    stop("`b` must be a single string.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Levenshtein edit distance
#'
#' Counts the minimum number of insertions, deletions and substitutions
#' needed to turn `a` into `b`, keeping only two rows of the cost table.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @family edit distances
#' @export
levenshteinDist = function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a = tolower(a)
    b = tolower(b)
  }
  ca = toChars(a)
  cb = toChars(b)
  prev = seq.int(0L, length(cb))
  for (i in seq_along(ca)) {
    curr = integer(length(cb) + 1L)
    curr[1] = i
    for (j in seq_along(cb)) {
      cost = if (ca[i] == cb[j]) 0L else 1L
      curr[j + 1L] = min(prev[j + 1L] + 1L, curr[j] + 1L, prev[j] + cost)
    }
    prev = curr
  }
  prev[length(prev)]
}

#' Optimal string alignment distance
#'
#' Like [levenshteinDist()] but also allows swapping two adjacent
#' characters at cost one, provided no substring is edited twice.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @family edit distances
#' @export
osaDist = function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a = tolower(a)
    b = tolower(b)
  }
  ca = toChars(a)
  cb = toChars(b)
  n = length(ca)
  m = length(cb)
  d = matrix(0L, n + 1L, m + 1L)
  d[, 1] = seq.int(0L, n)
  d[1, ] = seq.int(0L, m)
  for (i in seq_len(n)) {
    for (j in seq_len(m)) {
      cost = if (ca[i] == cb[j]) 0L else 1L
      d[i + 1L, j + 1L] = min(d[i, j + 1L] + 1L, d[i + 1L, j] + 1L, d[i, j] + cost)
      if (i > 1L && j > 1L && ca[i] == cb[j - 1L] && ca[i - 1L] == cb[j]) {
        d[i + 1L, j + 1L] = min(d[i + 1L, j + 1L], d[i - 1L, j - 1L] + 1L)
      }
    }
  }
  d[n + 1L, m + 1L]
}

#' Hamming distance
#'
#' Counts positions at which two equal-length strings differ.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @family edit distances
#' @export
hammingDist = function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a = tolower(a)
    b = tolower(b)
  }
  if (nchar(a) != nchar(b)) {
    stop("Hamming distance needs strings of equal length.", call. = FALSE)
  }
  sum(toChars(a) != toChars(b))
}

#' Longest common subsequence distance
#'
#' The number of characters that must be deleted from both strings so that
#' what remains is identical.
#'
#' @param a,b Single strings to compare.
#' @param ignoreCase Logical; compare case-insensitively?
#' @return A single integer distance, or `NA` if either input is missing.
#' @family edit distances
#' @export
lcsDist = function(a, b, ignoreCase = FALSE) {
  checkPair(a, b)
  if (is.na(a) || is.na(b)) {
    return(NA_integer_)
  }
  if (ignoreCase) {
    a = tolower(a)
    b = tolower(b)
  }
  ca = toChars(a)
  cb = toChars(b)
  prev = integer(length(cb) + 1L)
  for (i in seq_along(ca)) {
    curr = integer(length(cb) + 1L)
    for (j in seq_along(cb)) {
      curr[j + 1L] = if (ca[i] == cb[j]) prev[j] + 1L else max(prev[j + 1L], curr[j])
    }
    prev = curr
  }
  length(ca) + length(cb) - 2L * prev[length(prev)]
}

#' Edit distance by method name
#'
#' Vectorised front end: recycles `a` and `b` against each other and
#' applies the chosen distance to each pair.
#'
#' @param a,b Character vectors to compare element-wise.
#' @param method Which distance to compute.
#' @param ... Passed to the selected distance function.
#' @return An integer vector of distances.
#' @family edit distances
#' @export
stringDist = function(a, b, method = c("lv", "osa", "hamming", "lcs"), ...) {
  method = match.arg(method)
  fun = switch(method,
    lv = levenshteinDist,
    osa = osaDist,
    hamming = hammingDist,
    lcs = lcsDist
  )
  n = max(length(a), length(b))
  a = rep_len(a, n)
  b = rep_len(b, n)
  out = integer(n)
  for (i in seq_len(n)) {
    out[i] = fun(a[i], b[i], ...)
  }
  out
}

#' Normalised edit similarity
#'
#' Rescales a distance into a similarity between 0 and 1 by dividing by the
#' length of the longer string.
#'
#' @inheritParams stringDist
#' @return A numeric vector of similarities between 0 and 1.
#' @export
stringSim = function(a, b, method = c("lv", "osa", "lcs"), ...) {
  method = match.arg(method)
  d = stringDist(a, b, method = method, ...)
  maxLen = pmax(nchar(a), nchar(b))
  if (method == "lcs") {
    maxLen = nchar(a) + nchar(b)
  }
  ifelse(maxLen == 0, 1, 1 - d / maxLen)
}

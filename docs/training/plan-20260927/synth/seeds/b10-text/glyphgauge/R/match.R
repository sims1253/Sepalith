#' Normalise strings into comparison keys
#'
#' Lowercases, removes punctuation and sorts the remaining words, so that
#' "Smith, John" and "john smith" produce the same key.
#'
#' @param x A character vector of strings to normalise.
#' @param sortWords Logical; sort the words within each key?
#' @return A character vector of keys the same length as `x`.
#' @export
matchKey = function(x, sortWords = TRUE) {
  x = stringi::stri_trans_general(x, "Latin-ASCII")
  x = tolower(gsub("[[:punct:]]+", " ", x))
  words = strsplit(trimws(x), "\\s+")
  keys = vapply(words, function(w) {
    if (sortWords) w = sort(w)
    paste(w, collapse = " ")
  }, character(1))
  keys[is.na(x)] = NA_character_
  keys
}

#' Pairwise similarity matrix
#'
#' @param x,y Character vectors to compare; rows follow `x`, columns `y`.
#' @param simFun A function of two single strings returning a similarity.
#' @param ... Passed to `simFun`.
#' @return A numeric matrix of dimension `length(x)` by `length(y)`.
#' @family fuzzy matching
#' @export
simMatrix = function(x, y = x, simFun = jaccardSim, ...) {
  simFun = match.fun(simFun)
  out = matrix(NA_real_, length(x), length(y), dimnames = list(x, y))
  for (i in seq_along(x)) {
    for (j in seq_along(y)) {
      out[i, j] = simFun(x[i], y[j], ...)
    }
  }
  out
}

#' Find the closest candidate for each string
#'
#' @param x A character vector of strings to look up.
#' @param candidates A character vector of reference strings.
#' @param simFun A function of two single strings returning a similarity.
#' @param minSim Minimum similarity for a match to be accepted.
#' @param ... Passed to `simFun`.
#' @return A data frame with columns `query`, `match` and `similarity`;
#'   `match` is `NA` when nothing reaches `minSim`.
#' @family fuzzy matching
#' @export
closestMatch = function(x, candidates, simFun = jaccardSim, minSim = 0.5, ...) {
  sims = simMatrix(x, candidates, simFun = simFun, ...)
  best = apply(sims, 1, function(r) if (all(is.na(r))) NA_integer_ else which.max(r))
  bestSim = sims[cbind(seq_along(x), best)]
  matched = candidates[best]
  matched[is.na(bestSim) | bestSim < minSim] = NA_character_
  data.frame(
    query = x,
    match = matched,
    similarity = bestSim,
    stringsAsFactors = FALSE
  )
}

#' Group near-duplicate strings
#'
#' Greedy single-pass clustering: each string joins the first existing
#' group whose representative is at least `minSim` similar, otherwise it
#' starts a new group.
#'
#' @param x A character vector of strings to group.
#' @param simFun A function of two single strings returning a similarity.
#' @param minSim Minimum similarity for two strings to share a group.
#' @param ... Passed to `simFun`.
#' @return An integer vector of group ids the same length as `x`.
#' @family fuzzy matching
#' @export
groupNear = function(x, simFun = jaccardSim, minSim = 0.8, ...) {
  simFun = match.fun(simFun)
  groups = integer(length(x))
  reps = character()
  for (i in seq_along(x)) {
    assigned = FALSE
    for (g in seq_along(reps)) {
      if (simFun(x[i], reps[g], ...) >= minSim) {
        groups[i] = g
        assigned = TRUE
        break
      }
    }
    if (!assigned) {
      reps = c(reps, x[i])
      groups[i] = length(reps)
    }
  }
  groups
}

#' Validate e-mail addresses
#'
#' @param x A character vector of strings to validate.
#' @return A logical vector; `NA` inputs give `FALSE`.
#' @family validators
#' @export
isEmail = function(x) {
  pattern = paste0("^", extractPatterns()$email, "$")
  !is.na(x) & grepl(pattern, x, perl = TRUE)
}

#' Validate ticket identifiers
#'
#' @param x A character vector of strings to validate.
#' @return A logical vector; `NA` inputs give `FALSE`.
#' @family validators
#' @export
isTicketId = function(x) {
  pattern = paste0("^", extractPatterns()$ticketId, "$")
  !is.na(x) & grepl(pattern, x, perl = TRUE)
}

#' Validate ISO dates
#'
#' Checks the shape with a regular expression and then confirms the date
#' exists, so `2023-02-30` is rejected.
#'
#' @param x A character vector of strings to validate.
#' @return A logical vector; `NA` inputs give `FALSE`.
#' @family validators
#' @export
isIsoDate = function(x) {
  pattern = paste0("^", extractPatterns()$isoDate, "$")
  shaped = !is.na(x) & grepl(pattern, x, perl = TRUE)
  shaped & !is.na(as.Date(ifelse(shaped, x, NA_character_), format = "%Y-%m-%d"))
}

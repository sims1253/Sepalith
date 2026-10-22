#' @title Normalise strings into comparison keys
#'
#' @description Lowercases, removes punctuation and sorts the remaining
#'   words, so that "Smith, John" and "john smith" produce the same key.
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

#' @title Pairwise similarity matrix
#'
#' @param x,y Character vectors to compare; rows follow `x`, columns `y`.
#' @param simFun A function of two single strings returning a similarity.
#' @param ... Passed to `simFun`.
#' @return A numeric matrix of dimension `length(x)` by `length(y)`.
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

#' @title Find the closest candidate for each string
#'
#' @param x A character vector of strings to look up.
#' @param candidates A character vector of reference strings.
#' @param simFun A function of two single strings returning a similarity.
#' @param minSim Minimum similarity for a match to be accepted.
#' @param ... Passed to `simFun`.
#' @return A data frame with columns `query`, `match` and `similarity`;
#'   `match` is `NA` when nothing reaches `minSim`.
#' @export
closestMatch = function(x, candidates, simFun = jaccardSim, minSim = 0.5, ...) {
  if (length(candidates) == 0L) {
    return(data.frame(query = x, match = rep(NA_character_, length(x)),
      similarity = rep(NA_real_, length(x)), stringsAsFactors = FALSE))
  }

  sims = simMatrix(x, candidates, simFun = simFun, ...)
  # vapply() guarantees one integer index per query, even for no queries
  best = vapply(seq_along(x), function(i) {
    r = sims[i, ]
    if (all(is.na(r))) NA_integer_ else which.max(r)
  }, integer(1))
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

#' @title Group near-duplicate strings
#'
#' @description Greedy single-pass clustering: each string joins the first
#'   existing group whose representative is at least `minSim` similar,
#'   otherwise it starts a new group.
#'
#' @param x A character vector of strings to group.
#' @param simFun A function of two single strings returning a similarity.
#' @param minSim Minimum similarity for two strings to share a group.
#' @param ... Passed to `simFun`.
#' @return An integer vector of group ids the same length as `x`.
#' @export
groupNear = function(x, simFun = jaccardSim, minSim = 0.8, ...) {
  simFun = match.fun(simFun)
  groups = integer(length(x))
  reps = character()

  for (i in seq_along(x)) {
    assigned = FALSE
    for (g in seq_along(reps)) {
      # a missing similarity never joins a group
      if (isTRUE(simFun(x[i], reps[g], ...) >= minSim)) {
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

#' @title Validate strings against the extractor patterns
#'
#' @description `isEmail()` and `isTicketId()` anchor the patterns from the
#'   extractors so the whole string must match. `isIsoDate()` also confirms
#'   the date exists, so `2023-02-30` is rejected.
#'
#' @param x A character vector of strings to validate.
#' @return A logical vector; `NA` inputs give `FALSE`.
#' @rdname validators
#' @export
isEmail = function(x) {
  pattern = paste0("^", extractPatterns()$email, "$")
  !is.na(x) & grepl(pattern, x, perl = TRUE)
}

#' @rdname validators
#' @export
isTicketId = function(x) {
  pattern = paste0("^", extractPatterns()$ticketId, "$")
  !is.na(x) & grepl(pattern, x, perl = TRUE)
}

#' @rdname validators
#' @export
isIsoDate = function(x) {
  pattern = paste0("^", extractPatterns()$isoDate, "$")
  shaped = !is.na(x) & grepl(pattern, x, perl = TRUE)
  shaped & !is.na(as.Date(ifelse(shaped, x, NA_character_), format = "%Y-%m-%d"))
}

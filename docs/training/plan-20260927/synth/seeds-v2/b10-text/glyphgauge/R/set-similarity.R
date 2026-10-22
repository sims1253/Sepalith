#' @title Character q-grams of a string
#'
#' @description Pads the string with `q - 1` hash marks on each side so that
#'   the first and last characters take part in as many q-grams as the
#'   others.
#'
#' @param x A single string.
#' @param q Length of each q-gram.
#' @param pad Logical; pad the string before splitting?
#' @return A character vector of q-grams, possibly with repeats.
#' @export
qgrams = function(x, q = 2L, pad = TRUE) {
  if (is.na(x)) {
    return(NA_character_)
  }
  if (pad && q > 1L) {
    fill = strrep("#", q - 1L)
    x = paste0(fill, x, fill)
  }

  n = nchar(x)
  if (n < q) {
    return(character())
  }
  starts = seq_len(n - q + 1L)
  substring(x, starts, starts + q - 1L)
}

#' @title Set similarities of q-gram profiles
#'
#' @description `jaccardSim()` is the size of the intersection over the
#'   size of the union of the distinct q-grams; `diceSim()` is twice the
#'   intersection over the summed set sizes; `overlapSim()` divides by the
#'   smaller set, so a string is fully similar to any string containing it.
#'
#' @param a,b Single strings to compare.
#' @param q Length of each q-gram.
#' @return A number between 0 and 1, or `NA` if either input is missing.
#' @rdname setSimilarity
#' @export
jaccardSim = function(a, b, q = 2L) {
  if (is.na(a) || is.na(b)) {
    return(NA_real_)
  }
  sa = unique(qgrams(a, q))
  sb = unique(qgrams(b, q))

  total = length(union(sa, sb))
  if (total == 0L) {
    return(1)
  }
  length(intersect(sa, sb)) / total
}

#' @rdname setSimilarity
#' @export
diceSim = function(a, b, q = 2L) {
  if (is.na(a) || is.na(b)) {
    return(NA_real_)
  }
  sa = unique(qgrams(a, q))
  sb = unique(qgrams(b, q))

  total = length(sa) + length(sb)
  if (total == 0L) {
    return(1)
  }
  2 * length(intersect(sa, sb)) / total
}

#' @rdname setSimilarity
#' @export
overlapSim = function(a, b, q = 2L) {
  if (is.na(a) || is.na(b)) {
    return(NA_real_)
  }
  sa = unique(qgrams(a, q))
  sb = unique(qgrams(b, q))

  smaller = min(length(sa), length(sb))
  if (smaller == 0L) {
    # two empty profiles are identical; empty against non-empty shares nothing
    return(as.numeric(length(sa) == length(sb)))
  }
  length(intersect(sa, sb)) / smaller
}

#' @title Cosine similarity of q-gram count profiles
#'
#' @description Unlike the set-based measures this one counts repeated
#'   q-grams.
#'
#' @param a,b Single strings to compare.
#' @param q Length of each q-gram.
#' @return A number between 0 and 1, or `NA` if either input is missing.
#' @export
qgramCosine = function(a, b, q = 2L) {
  if (is.na(a) || is.na(b)) {
    return(NA_real_)
  }
  ga = qgrams(a, q)
  gb = qgrams(b, q)

  # count vectors over a shared vocabulary so the positions line up
  keys = union(ga, gb)
  va = table(factor(ga, levels = keys)) |> as.numeric()
  vb = table(factor(gb, levels = keys)) |> as.numeric()

  denom = sqrt(sum(va^2)) * sqrt(sum(vb^2))
  if (denom == 0) {
    return(1)
  }
  sum(va * vb) / denom
}

#' @title Jaro-Winkler similarity
#'
#' @description Counts characters that match within a sliding window,
#'   penalises transpositions among them, and then boosts pairs that share
#'   a prefix.
#'
#' @param a,b Single strings to compare.
#' @param prefixWeight Boost applied per shared leading character, at most
#'   0.25.
#' @param maxPrefix Maximum number of leading characters that earn a boost.
#' @return A number between 0 and 1, or `NA` if either input is missing.
#' @export
jaroWinklerSim = function(a, b, prefixWeight = 0.1, maxPrefix = 4L) {
  if (is.na(a) || is.na(b)) {
    return(NA_real_)
  }
  ca = strsplit(a, "")[[1]]
  cb = strsplit(b, "")[[1]]
  la = length(ca)
  lb = length(cb)
  if (la == 0L && lb == 0L) {
    return(1)
  }

  # characters only count as matching within this distance of each other
  window = max(floor(max(la, lb) / 2) - 1, 0)
  usedB = logical(lb)
  matchA = logical(la)
  for (i in seq_len(la)) {
    lo = max(1, i - window)
    hi = min(lb, i + window)
    if (lo > hi) next
    for (j in lo:hi) {
      if (!usedB[j] && ca[i] == cb[j]) {
        usedB[j] = TRUE
        matchA[i] = TRUE
        break
      }
    }
  }

  m = sum(matchA)
  if (m == 0) {
    return(0)
  }
  # matched characters taken in order from each side; mismatches are swaps
  halfT = sum(ca[matchA] != cb[usedB]) / 2
  jaro = (m / la + m / lb + (m - halfT) / m) / 3

  prefix = 0
  for (k in seq_len(min(maxPrefix, la, lb))) {
    if (ca[k] != cb[k]) break
    prefix = prefix + 1
  }
  jaro + prefix * min(prefixWeight, 0.25) * (1 - jaro)
}

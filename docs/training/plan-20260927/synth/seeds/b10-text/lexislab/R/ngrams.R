#' Build word n-grams from a single token vector
#'
#' @param tokens A character vector of tokens for one document.
#' @param n Size of each n-gram.
#' @param sep Separator placed between the words of an n-gram.
#' @return A character vector of n-grams; empty when there are fewer than
#'   `n` tokens.
#' @noRd
ngrams_one <- function(tokens, n, sep) {
  len <- length(tokens)
  if (len < n) {
    return(character())
  }
  starts <- seq_len(len - n + 1L)
  vapply(starts, function(i) {
    paste(tokens[i:(i + n - 1L)], collapse = sep)
  }, character(1))
}

#' Build word n-grams for every document
#'
#' Applies a sliding window of `n` tokens over each document.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param n Size of each n-gram.
#' @param sep Separator placed between the words of an n-gram.
#' @return A list of character vectors, one element per document.
#' @family n-gram helpers
#' @export
make_ngrams <- function(tokens, n = 2L, sep = " ") {
  if (length(n) != 1L || is.na(n) || n < 1L) {
    stop("`n` must be a single positive integer.", call. = FALSE)
  }
  out <- lapply(tokens, ngrams_one, n = as.integer(n), sep = sep)
  names(out) <- names(tokens)
  out
}

#' Build all n-grams within a size range
#'
#' Concatenates the n-grams of every size from `n_min` to `n_max` for
#' each document, smallest size first.
#'
#' @inheritParams make_ngrams
#' @param n_min Smallest n-gram size.
#' @param n_max Largest n-gram size.
#' @return A list of character vectors, one element per document.
#' @family n-gram helpers
#' @export
make_ngram_range <- function(tokens, n_min = 1L, n_max = 3L, sep = " ") {
  if (n_min > n_max) {
    stop("`n_min` must not exceed `n_max`.", call. = FALSE)
  }
  sizes <- seq.int(n_min, n_max)
  out <- vector("list", length(tokens))
  for (i in seq_along(tokens)) {
    out[[i]] <- unlist(lapply(sizes, function(n) ngrams_one(tokens[[i]], n, sep)))
    if (is.null(out[[i]])) {
      out[[i]] <- character()
    }
  }
  names(out) <- names(tokens)
  out
}

#' Build skip-grams for every document
#'
#' A skip-gram pairs each token with the tokens up to `k` positions ahead
#' of it, allowing gaps between the words.
#'
#' @inheritParams make_ngrams
#' @param k Maximum number of tokens that may be skipped.
#' @return A list of character vectors, one element per document.
#' @family n-gram helpers
#' @export
make_skipgrams <- function(tokens, k = 1L, sep = " ") {
  out <- lapply(tokens, function(tok) {
    len <- length(tok)
    pairs <- character()
    for (i in seq_len(max(len - 1L, 0L))) {
      ahead <- seq.int(i + 1L, min(i + k + 1L, len))
      pairs <- c(pairs, paste(tok[i], tok[ahead], sep = sep))
    }
    pairs
  })
  names(out) <- names(tokens)
  out
}

#' Count n-gram frequencies across a corpus
#'
#' @param ngrams A list of character vectors as returned by [make_ngrams()].
#' @param min_count Drop n-grams seen fewer than this many times.
#' @return A data frame with columns `ngram` and `count`, sorted by
#'   decreasing count.
#' @family n-gram helpers
#' @export
count_ngrams <- function(ngrams, min_count = 1L) {
  flat <- unlist(ngrams, use.names = FALSE)
  if (length(flat) == 0L) {
    return(data.frame(ngram = character(), count = integer()))
  }
  tab <- table(flat)
  tab <- tab[tab >= min_count]
  out <- data.frame(
    ngram = names(tab),
    count = as.integer(tab),
    stringsAsFactors = FALSE
  )
  out <- out[order(-out$count, out$ngram), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Score bigrams by pointwise mutual information
#'
#' Compares how often two words appear next to each other with how often
#' they would co-occur by chance given their individual frequencies.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param min_count Drop bigrams seen fewer than this many times.
#' @return A data frame with columns `first`, `second`, `count` and `pmi`,
#'   sorted by decreasing `pmi`.
#' @export
bigram_pmi <- function(tokens, min_count = 2L) {
  unigrams <- table(unlist(tokens, use.names = FALSE))
  bigrams <- count_ngrams(make_ngrams(tokens, n = 2L, sep = "\t"), min_count)
  if (nrow(bigrams) == 0L) {
    return(data.frame(
      first = character(), second = character(),
      count = integer(), pmi = numeric()
    ))
  }
  parts <- do.call(rbind, strsplit(bigrams$ngram, "\t", fixed = TRUE))
  n_uni <- sum(unigrams)
  n_bi <- sum(bigrams$count)
  p_xy <- bigrams$count / n_bi
  p_x <- as.numeric(unigrams[parts[, 1]]) / n_uni
  p_y <- as.numeric(unigrams[parts[, 2]]) / n_uni
  out <- data.frame(
    first = parts[, 1],
    second = parts[, 2],
    count = bigrams$count,
    pmi = log2(p_xy / (p_x * p_y)),
    stringsAsFactors = FALSE
  )
  out[order(-out$pmi), , drop = FALSE]
}

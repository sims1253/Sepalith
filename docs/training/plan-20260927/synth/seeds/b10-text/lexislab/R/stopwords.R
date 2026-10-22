#' Built-in stop-word lists
#'
#' Returns one of the small stop-word lists shipped with lexislab.
#'
#' @param language Which language list to return.
#' @return A sorted character vector of lowercase stop words.
#' @family stop-word helpers
#' @export
stopword_list <- function(language = c("en", "de", "minimal")) {
  language <- match.arg(language)
  words <- switch(language,
    en = c(
      "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
      "has", "have", "he", "in", "is", "it", "its", "of", "on", "or", "she",
      "that", "the", "their", "they", "this", "to", "was", "were", "will", "with"
    ),
    de = c(
      "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "das", "dass",
      "dem", "den", "der", "die", "ein", "eine", "einen", "er", "es", "fuer",
      "hat", "im", "in", "ist", "mit", "nicht", "noch", "sie", "und", "von",
      "war", "wie", "zu"
    ),
    minimal = c("a", "an", "and", "of", "the", "to")
  )
  sort(unique(words))
}

#' Remove stop words from tokenised documents
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param stopwords A character vector of words to drop.
#' @param ignore_case Logical; compare in lowercase?
#' @return A list of character vectors, one element per document.
#' @family stop-word helpers
#' @export
remove_stopwords <- function(tokens, stopwords = stopword_list("en"),
                             ignore_case = TRUE) {
  if (ignore_case) {
    stopwords <- tolower(stopwords)
  }
  out <- lapply(tokens, function(tok) {
    key <- if (ignore_case) tolower(tok) else tok
    tok[!key %in% stopwords]
  })
  names(out) <- names(tokens)
  out
}

#' Keep only words from a vocabulary
#'
#' The complement of [remove_stopwords()]: tokens not found in `keep` are
#' dropped.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param keep A character vector of words to retain.
#' @param ignore_case Logical; compare in lowercase?
#' @return A list of character vectors, one element per document.
#' @family stop-word helpers
#' @export
keep_vocabulary <- function(tokens, keep, ignore_case = TRUE) {
  if (ignore_case) {
    keep <- tolower(keep)
  }
  out <- lapply(tokens, function(tok) {
    key <- if (ignore_case) tolower(tok) else tok
    tok[key %in% keep]
  })
  names(out) <- names(tokens)
  out
}

#' Derive stop words from corpus frequency
#'
#' Treats any term appearing in more than `max_doc_prop` of documents as a
#' stop word, which catches boilerplate specific to a corpus.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param max_doc_prop Maximum proportion of documents a term may appear in.
#' @param min_docs Only consider corpora with at least this many documents.
#' @return A sorted character vector of corpus-specific stop words.
#' @family stop-word helpers
#' @export
corpus_stopwords <- function(tokens, max_doc_prop = 0.8, min_docs = 5L) {
  n_docs <- length(tokens)
  if (n_docs < min_docs) {
    warning("Too few documents to derive stop words; returning none.",
      call. = FALSE
    )
    return(character())
  }
  doc_freq <- table(unlist(lapply(tokens, unique), use.names = FALSE))
  common <- names(doc_freq)[doc_freq / n_docs > max_doc_prop]
  sort(common)
}

#' Strip a fixed set of English suffixes
#'
#' A deliberately light stemmer: it removes the longest matching suffix
#' while keeping at least `min_stem` characters of the word.
#'
#' @param words A character vector of lowercase words.
#' @param min_stem Minimum number of characters to leave in the stem.
#' @return A character vector of stems the same length as `words`.
#' @export
light_stem <- function(words, min_stem = 3L) {
  suffixes <- c("ational", "ization", "ness", "ment", "ing", "ies", "ed", "ly", "es", "s")
  stems <- words
  for (i in seq_along(words)) {
    w <- words[i]
    if (is.na(w)) {
      next
    }
    for (suf in suffixes) {
      cut <- nchar(w) - nchar(suf)
      if (cut >= min_stem && endsWith(w, suf)) {
        stems[i] <- substr(w, 1L, cut)
        break
      }
    }
  }
  stems
}

#' Apply a word-level function to every document
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param fun A function taking and returning a character vector.
#' @param ... Passed to `fun`.
#' @return A list of character vectors, one element per document.
#' @keywords internal
map_tokens <- function(tokens, fun, ...) {
  fun <- match.fun(fun)
  out <- lapply(tokens, fun, ...)
  names(out) <- names(tokens)
  out
}

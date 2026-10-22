#' Normalise raw text before tokenising
#'
#' @description Lowercases, strips control characters and collapses runs of
#'   whitespace. Missing values are kept as `NA` so positions line up with
#'   the input.
#'
#' @param text A character vector of documents.
#' @param lower Logical; should the text be lowercased?
#' @param strip_digits Logical; should digits be removed?
#' @return A character vector the same length as `text`.
#' @export
normalise_text <- function(text, lower = TRUE, strip_digits = FALSE) {
  if (!is.character(text)) {
    stop("`text` must be a character vector.", call. = FALSE)
  }

  out <- gsub("[[:cntrl:]]+", " ", text)
  if (lower) {
    out <- tolower(out)
  }
  if (strip_digits) {
    out <- gsub("[[:digit:]]+", " ", out)
  }

  # collapse last, so removed characters do not leave double spaces behind
  out <- gsub("[[:space:]]+", " ", out)
  out <- trimws(out)
  return(out)
}

#' Split documents into word tokens
#'
#' @description Splits each document on anything that is not a letter,
#'   digit or apostrophe. Empty tokens are dropped, and a missing document
#'   gives an empty token vector.
#'
#' @param text A character vector of documents.
#' @param min_length Minimum number of characters a token must have.
#' @return A list of character vectors, one element per document.
#' @export
tokenise_words <- function(text, min_length = 1L) {
  text <- normalise_text(text)
  pieces <- strsplit(text, "[^[:alnum:]']+", perl = TRUE)

  tokens <- lapply(pieces, function(p) {
    p <- p[!is.na(p) & nzchar(p)]
    p[nchar(p) >= min_length]
  })
  names(tokens) <- names(text)
  return(tokens)
}

#' Split documents into sentences
#'
#' @description Uses a simple terminal-punctuation rule: a sentence ends at
#'   `.`, `!` or `?` followed by whitespace and an uppercase letter or end of
#'   string.
#'
#' @param text A character vector of documents.
#' @param min_length Minimum number of characters a sentence must have.
#' @return A list of character vectors, one element per document.
#' @export
tokenise_sentences <- function(text, min_length = 1L) {
  text <- gsub("[[:space:]]+", " ", text)
  # lookarounds keep the punctuation with its sentence
  pieces <- strsplit(text, "(?<=[.!?])\\s+(?=[A-Z])", perl = TRUE)

  tokens <- lapply(pieces, function(p) {
    p <- trimws(p[!is.na(p)])
    p[nchar(p) >= min_length]
  })
  names(tokens) <- names(text)
  return(tokens)
}

#' Split documents into character shingles
#'
#' @description Produces overlapping character windows of width `k`.
#'   Whitespace is collapsed to a single underscore so word boundaries
#'   remain visible.
#'
#' @param text A character vector of documents.
#' @param k Width of each shingle in characters.
#' @return A list of character vectors, one element per document.
#' @export
tokenise_shingles <- function(text, k = 3L) {
  if (length(k) != 1L || is.na(k) || k < 1L) {
    stop("`k` must be a single positive integer.", call. = FALSE)
  }
  text <- gsub(" ", "_", normalise_text(text), fixed = TRUE)

  tokens <- lapply(text, function(doc) {
    n <- nchar(doc)
    if (is.na(doc) || n < k) {
      return(character())
    }
    starts <- seq_len(n - k + 1L)
    substring(doc, starts, starts + k - 1L)
  })
  names(tokens) <- names(text)
  return(tokens)
}

#' Tokenise with a named method
#'
#' @description Dispatches to one of the tokenisers in this package by
#'   name.
#'
#' @param text A character vector of documents.
#' @param method Which tokeniser to use.
#' @param ... Passed to the selected tokeniser.
#' @return A list of character vectors, one element per document.
#' @export
tokenise <- function(text, method = c("words", "sentences", "shingles"), ...) {
  method <- match.arg(method)
  tokens <- switch(method,
    words = tokenise_words(text, ...),
    sentences = tokenise_sentences(text, ...),
    shingles = tokenise_shingles(text, ...)
  )
  return(tokens)
}

#' Count tokens per document
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @keywords internal
count_tokens <- function(tokens) {
  counts <- vapply(tokens, length, integer(1))
  names(counts) <- names(tokens)
  return(counts)
}

#' Flatten a token list into a long data frame
#'
#' @description Each row holds one token with the document it came from
#'   and its position within that document.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param doc_ids Optional identifiers for the documents. Defaults to the
#'   names of `tokens`, or sequential integers when there are none.
#' @return A data frame with columns `doc_id`, `position` and `token`.
#' @export
tokens_to_long <- function(tokens, doc_ids = names(tokens)) {
  if (is.null(doc_ids)) {
    doc_ids <- as.character(seq_along(tokens))
  }
  if (length(doc_ids) != length(tokens)) {
    stop("`doc_ids` must have one entry per document.", call. = FALSE)
  }

  lens <- count_tokens(tokens)
  out <- data.frame(
    doc_id = rep(doc_ids, lens),
    position = sequence(lens),
    token = as.character(unlist(tokens, use.names = FALSE)),
    stringsAsFactors = FALSE
  )
  rownames(out) <- NULL
  return(out)
}

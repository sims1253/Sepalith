#' Build a term-document matrix by hand
#'
#' @description Counts how often each vocabulary term occurs in each
#'   document. Terms are rows and documents are columns.
#'
#' @param tokens A list of character vectors as returned by [tokenise()].
#' @param vocabulary Optional character vector fixing the row order. Terms
#'   not in the vocabulary are ignored.
#' @return An integer matrix with terms as rows and documents as columns.
#' @export
build_tdm <- function(tokens, vocabulary = NULL) {
  if (is.null(vocabulary)) {
    vocabulary <- sort(unique(as.character(unlist(tokens, use.names = FALSE))))
  }
  doc_ids <- names(tokens)
  if (is.null(doc_ids)) {
    doc_ids <- paste0("doc", seq_along(tokens))
  }

  tdm <- matrix(0L,
    nrow = length(vocabulary), ncol = length(tokens),
    dimnames = list(term = vocabulary, doc = doc_ids)
  )
  for (j in seq_along(tokens)) {
    hits <- tokens[[j]][tokens[[j]] %in% vocabulary]
    if (length(hits) == 0L) {
      next
    }
    counts <- table(hits)
    tdm[names(counts), j] <- as.integer(counts)
  }

  return(tdm)
}

#' Term-frequency weighting
#'
#' @param tdm A term-document count matrix as returned by [build_tdm()].
#' @param scheme How to scale raw counts.
#' @return A numeric matrix with the same dimensions as `tdm`.
#' @family weighting
#' @export
weight_tf <- function(tdm, scheme = c("raw", "proportion", "log", "binary")) {
  scheme <- match.arg(scheme)
  # pmax() keeps empty documents at zero instead of 0/0
  out <- switch(scheme,
    raw = tdm * 1,
    proportion = sweep(tdm, 2, pmax(colSums(tdm), 1), "/"),
    log = log1p(tdm),
    binary = (tdm > 0) * 1
  )
  return(out)
}

#' Inverse document frequency
#'
#' @param tdm A term-document count matrix as returned by [build_tdm()].
#' @param smooth Logical; add one to document counts to avoid division by
#'   zero and to dampen very rare terms?
#' @return A named numeric vector with one weight per term.
#' @family weighting
#' @export
weight_idf <- function(tdm, smooth = TRUE) {
  n_docs <- ncol(tdm)
  doc_freq <- rowSums(tdm > 0)
  if (smooth) {
    idf <- log((1 + n_docs) / (1 + doc_freq)) + 1
  } else {
    idf <- log(n_docs / doc_freq)
  }
  names(idf) <- rownames(tdm)
  return(idf)
}

#' TF-IDF weighting
#'
#' @description Multiplies a term-frequency matrix by the inverse document
#'   frequency of each term, optionally normalising each document to unit
#'   length.
#'
#' @param tdm A term-document count matrix as returned by [build_tdm()].
#' @param tf Term-frequency scheme passed to [weight_tf()].
#' @param smooth Passed to [weight_idf()].
#' @param normalise Logical; scale each document column to unit L2 norm?
#' @return A numeric matrix with the same dimensions as `tdm`.
#' @family weighting
#' @export
weight_tfidf <- function(tdm, tf = "raw", smooth = TRUE, normalise = TRUE) {
  tf_mat <- weight_tf(tdm, scheme = tf)
  idf <- weight_idf(tdm, smooth = smooth)
  # idf has one entry per row, so plain recycling scales each term
  out <- tf_mat * idf

  if (normalise) {
    norms <- sqrt(colSums(out^2))
    norms[norms == 0] <- 1
    out <- sweep(out, 2, norms, "/")
  }
  return(out)
}

#' Okapi BM25 weighting
#'
#' @description Saturates term frequency with `k1` and penalises long
#'   documents with `b`, using the same smoothed IDF as [weight_idf()].
#'
#' @param tdm A term-document count matrix as returned by [build_tdm()].
#' @param k1 Term-frequency saturation parameter.
#' @param b Document-length normalisation parameter between 0 and 1.
#' @return A numeric matrix with the same dimensions as `tdm`.
#' @export
weight_bm25 <- function(tdm, k1 = 1.2, b = 0.75) {
  if (b < 0 || b > 1) {
    stop("`b` must lie between 0 and 1.", call. = FALSE)
  }

  doc_len <- colSums(tdm)
  avg_len <- mean(doc_len)
  if (is.na(avg_len) || avg_len == 0) {
    # an empty corpus has no length to normalise against
    avg_len <- 1
  }
  idf <- weight_idf(tdm, smooth = TRUE)
  denom_doc <- k1 * (1 - b + b * doc_len / avg_len)

  out <- tdm * 0
  for (j in seq_len(ncol(tdm))) {
    f <- tdm[, j]
    out[, j] <- idf * f * (k1 + 1) / (f + denom_doc[j])
  }
  return(out)
}

#' Cosine similarity between documents
#'
#' @param weights A weighted term-document matrix, e.g. from
#'   [weight_tfidf()].
#' @return A symmetric document-by-document similarity matrix.
#' @export
doc_cosine <- function(weights) {
  norms <- sqrt(colSums(weights^2))
  norms[norms == 0] <- 1
  unit <- sweep(weights, 2, norms, "/")
  sim <- crossprod(unit)

  # an empty document is not similar to anything, itself included
  diag(sim)[colSums(weights^2) == 0] <- 0
  return(sim)
}

#' Highest-weighted terms per document
#'
#' @param weights A weighted term-document matrix, e.g. from
#'   [weight_tfidf()].
#' @param n Number of terms to return for each document.
#' @return A data frame with columns `doc`, `rank`, `term` and `weight`.
#' @export
top_terms <- function(weights, n = 5L) {
  docs <- colnames(weights)
  pieces <- vector("list", length(docs))

  for (j in seq_along(docs)) {
    w <- weights[, j]
    w <- w[w > 0]
    # ties are broken alphabetically so the output is stable
    keep <- head(order(-w, names(w)), n)
    pieces[[j]] <- data.frame(
      doc = rep(docs[j], length(keep)),
      rank = seq_along(keep),
      term = names(w)[keep],
      weight = unname(w[keep]),
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, pieces)
  rownames(out) <- NULL
  return(out)
}

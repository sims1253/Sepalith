#' Clean text with a sequence of rules
#'
#' @description
#' Rules run in the order given in `steps`. Every rule replaces its matches
#' with `replacement`, except `"space"`, which collapses white space.
#'
#' @param x A character vector of documents.
#' @param steps The cleaning rules to apply.
#' @param replacement Text inserted in place of each removed match.
#' @return A character vector the same length as `x`, with an attribute
#'   `removed` giving the number of matches per rule.
#' @export
cleanText = function(x, steps = c("tags", "urls", "digits", "punct", "space"),
                     replacement = " ") {
  steps = match.arg(steps, several.ok = TRUE)
  removed = integer()
  # Each rule counts its matches, then replaces them
  for (step in steps) {
    if (step == "tags") {
      pattern = "<[^>]*>"
      removed[["tags"]] = sum(lengths(regmatches(x, gregexpr(pattern, x))))
      x = gsub(pattern, replacement, x)
    }
    if (step == "urls") {
      pattern = "https?://[^[:space:]]+"
      removed[["urls"]] = sum(lengths(regmatches(x, gregexpr(pattern, x))))
      x = gsub(pattern, replacement, x)
    }
    if (step == "digits") {
      pattern = "[[:digit:]]+"
      removed[["digits"]] = sum(lengths(regmatches(x, gregexpr(pattern, x))))
      x = gsub(pattern, replacement, x)
    }
    if (step == "punct") {
      pattern = "[[:punct:]]+"
      removed[["punct"]] = sum(lengths(regmatches(x, gregexpr(pattern, x))))
      x = gsub(pattern, replacement, x)
    }
    if (step == "space") {
      pattern = "[[:space:]]+"
      removed[["space"]] = sum(lengths(regmatches(x, gregexpr("[[:space:]]{2,}", x))))
      x = trimws(gsub(pattern, " ", x))
    }
  }

  attr(x, "removed") = removed
  x
}

#' Word n-grams
#'
#' @param tokens A character vector of tokens from one document.
#' @param n The n-gram length.
#' @param sep String placed between the tokens of an n-gram.
#' @return A character vector of `length(tokens) - n + 1` n-grams, or an
#'   empty vector when there are fewer than `n` tokens.
#' @export
wordNgrams = function(tokens, n = 2L, sep = " ") {
  count = length(tokens) - n + 1L
  if (count < 1L) {
    return(character())
  }
  starts = seq_len(count)
  vapply(starts, function(i) paste(tokens[i:(i + n - 1L)], collapse = sep), character(1))
}

#' Character n-grams
#'
#' @param text A single string.
#' @param n The n-gram length.
#' @param pad Character added `n - 1` times at both ends, or `""` for none.
#' @return A character vector of n-grams, or an empty vector when the padded
#'   text is shorter than `n`.
#' @export
charNgrams = function(text, n = 3L, pad = "_") {
  text = paste0(strrep(pad, n - 1L), text, strrep(pad, n - 1L))
  count = nchar(text) - n + 1L
  if (count < 1L) {
    return(character())
  }
  starts = seq_len(count)
  substring(text, starts, starts + n - 1L)
}

#' Keywords in context
#'
#' @param x A character vector of documents.
#' @param keyword A regular expression matched against single words.
#' @param before Number of words shown to the left of each match.
#' @param after Number of words shown to the right of each match.
#' @param ignoreCase Match `keyword` regardless of case?
#' @return A data frame with columns `doc`, `position`, `left`, `keyword` and
#'   `right`, one row per keyword occurrence.
#' @export
kwic = function(x, keyword, before = 3L, after = before, ignoreCase = FALSE) {
  if (before < 0 || after < 0) {
    stop("`before` and `after` must not be negative.", call. = FALSE)
  }
  rows = list()
  for (i in seq_along(x)) {
    words = strsplit(trimws(x[[i]]), "[[:space:]]+")[[1]]
    hits = grep(keyword, words, perl = TRUE, ignore.case = ignoreCase)
    for (h in hits) {
      # Words to the left
      leftIdx = seq_len(h - 1L)
      leftIdx = utils::tail(leftIdx, before)
      left = paste(words[leftIdx], collapse = " ")

      # Words to the right
      rightIdx = seq_len(length(words) - h) + h
      rightIdx = utils::head(rightIdx, after)
      right = paste(words[rightIdx], collapse = " ")

      rows[[length(rows) + 1L]] = data.frame(
        doc = i,
        position = h,
        left = left,
        keyword = words[h],
        right = right
      )
    }
  }
  if (length(rows) == 0L) {
    return(data.frame(doc = integer(), position = integer(), left = character(),
                      keyword = character(), right = character()))
  }
  do.call(rbind, rows)
}

#' Term weights across documents
#'
#' @param tokens A list with one character vector of tokens per document.
#' @param weight The weighting: raw `"count"`, `"binary"` presence, `"tf"`
#'   (count divided by document length) or `"tfidf"` (tf times the natural
#'   log of the inverse document frequency).
#' @param minCount Drop terms whose total count is below this value.
#' @return A numeric matrix with one row per document and one column per
#'   term, columns sorted alphabetically.
#' @export
termMatrix = function(tokens, weight = c("count", "binary", "tf", "tfidf"), minCount = 1L) {
  weight = match.arg(weight)
  if (!is.list(tokens)) {
    stop("`tokens` must be a list of character vectors.", call. = FALSE)
  }

  terms = sort(unique(unlist(tokens, use.names = FALSE)))
  counts = matrix(0, nrow = length(tokens), ncol = length(terms),
                  dimnames = list(names(tokens), terms))
  for (i in seq_along(tokens)) {
    tab = table(tokens[[i]])
    counts[i, names(tab)] = as.numeric(tab)
  }

  counts = counts[, colSums(counts) >= minCount, drop = FALSE]
  docLength = pmax(rowSums(counts), 1)

  switch(weight,
    count = {
      out = counts
    },
    binary = {
      out = counts
      out[] = as.numeric(counts > 0)
    },
    tf = {
      out = counts / docLength
    },
    tfidf = {
      docFreq = colSums(counts > 0)
      idf = log(nrow(counts) / pmax(docFreq, 1))
      out = sweep(counts / docLength, 2L, idf, "*")
    }
  )
  out
}

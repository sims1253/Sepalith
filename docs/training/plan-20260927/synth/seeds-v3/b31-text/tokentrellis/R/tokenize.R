#' Token patterns
#'
#' @description
#' The regular expressions behind [tokenizeOne()]. Earlier rules win when
#' two matches overlap.
#'
#' @return A named list of PCRE patterns with elements `url`, `number`,
#'   `word` and `punct`.
#' @noRd
tokenRules = function() {
  list(
    url = "https?://[^\\s]+[^\\s.,;:!?)]",
    number = "[-+]?[0-9]+(?:[.,][0-9]+)*",
    word = "[[:alpha:]]+(?:['-][[:alpha:]]+)*",
    punct = "[[:punct:]]"
  )
}

#' Locate the matches of one pattern
#'
#' @param text A single string.
#' @param pattern A regular expression.
#' @param type The token type recorded for every match.
#' @return A data frame with columns `start`, `end` and `type`.
#' @noRd
matchSpans = function(text, pattern, type) {
  m = gregexpr(pattern, text, perl = TRUE)[[1]]
  if (m[1] < 0) {
    return(data.frame(start = integer(), end = integer(), type = character()))
  }
  data.frame(
    start = as.integer(m),
    end = as.integer(m) + attr(m, "match.length") - 1L,
    type = type
  )
}

#' Tokenize a single string
#'
#' @description
#' Applies each rule in turn, drops matches that overlap a match of an
#' earlier rule and returns the tokens in reading order.
#'
#' @param text A single string.
#' @param rules The token rules to apply, in priority order.
#' @param lower Convert `word` tokens to lower case?
#' @return A data frame with columns `token`, `type`, `start` and `end`.
#' @export
tokenizeOne = function(text, rules = c("url", "number", "word", "punct"), lower = FALSE) {
  rules = match.arg(rules, several.ok = TRUE)
  patterns = tokenRules()
  taken = logical(nchar(text))
  spans = data.frame(start = integer(), end = integer(), type = character())

  # Rule `url`
  if ("url" %in% rules) {
    found = matchSpans(text, patterns$url, "url")
    keep = vapply(seq_len(nrow(found)), function(i) {
      !any(taken[found$start[i]:found$end[i]])
    }, logical(1))
    found = found[keep, , drop = FALSE]
    for (i in seq_len(nrow(found))) taken[found$start[i]:found$end[i]] = TRUE
    spans = rbind(spans, found)
  }

  # Rule `number`
  if ("number" %in% rules) {
    found = matchSpans(text, patterns$number, "number")
    keep = vapply(seq_len(nrow(found)), function(i) {
      !any(taken[found$start[i]:found$end[i]])
    }, logical(1))
    found = found[keep, , drop = FALSE]
    for (i in seq_len(nrow(found))) taken[found$start[i]:found$end[i]] = TRUE
    spans = rbind(spans, found)
  }

  # Rule `word`
  if ("word" %in% rules) {
    found = matchSpans(text, patterns$word, "word")
    keep = vapply(seq_len(nrow(found)), function(i) {
      !any(taken[found$start[i]:found$end[i]])
    }, logical(1))
    found = found[keep, , drop = FALSE]
    for (i in seq_len(nrow(found))) taken[found$start[i]:found$end[i]] = TRUE
    spans = rbind(spans, found)
  }

  # Rule `punct`
  if ("punct" %in% rules) {
    found = matchSpans(text, patterns$punct, "punct")
    keep = vapply(seq_len(nrow(found)), function(i) {
      !any(taken[found$start[i]:found$end[i]])
    }, logical(1))
    found = found[keep, , drop = FALSE]
    for (i in seq_len(nrow(found))) taken[found$start[i]:found$end[i]] = TRUE
    spans = rbind(spans, found)
  }

  spans = spans[order(spans$start), , drop = FALSE]
  token = substr(rep(text, nrow(spans)), spans$start, spans$end)
  if (lower) {
    token[spans$type == "word"] = tolower(token[spans$type == "word"])
  }
  data.frame(token = token, type = spans$type, start = spans$start, end = spans$end)
}

#' Tokenize a character vector
#'
#' @param x A character vector of documents. Names, when present, are used
#'   as document identifiers.
#' @param rules The token rules to apply, in priority order.
#' @param lower Convert `word` tokens to lower case?
#' @param drop Token types to leave out of the result, for example `"punct"`.
#' @return A data frame with columns `doc`, `id`, `token`, `type`, `start`
#'   and `end`.
#' @export
tokenize = function(x, rules = c("url", "number", "word", "punct"), lower = FALSE,
                    drop = character()) {
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }
  ids = names(x)
  if (is.null(ids)) {
    ids = as.character(seq_along(x))
  }
  x[is.na(x)] = ""

  pieces = lapply(seq_along(x), function(i) {
    tok = tokenizeOne(x[[i]], rules = rules, lower = lower)
    tok = tok[!tok$type %in% drop, , drop = FALSE]
    cbind(doc = rep(i, nrow(tok)), id = rep(ids[i], nrow(tok)), tok)
  })

  out = do.call(rbind, pieces)
  rownames(out) = NULL
  out
}

#' Count units of text
#'
#' @param x A character vector of documents.
#' @param by The unit to count. Paragraphs are separated by blank lines.
#' @return An integer vector the same length as `x`; missing documents count
#'   as empty.
#' @export
countUnits = function(x, by = c("word", "sentence", "line", "paragraph", "char")) {
  by = match.arg(by)
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }
  x[is.na(x)] = ""

  switch(by,
    word = {
      parts = strsplit(trimws(x), "[[:space:]]+")
      parts = lapply(parts, function(p) p[p != ""])
      lengths(parts)
    },
    sentence = {
      parts = strsplit(trimws(x), "(?<=[.!?])\\s+", perl = TRUE)
      parts = lapply(parts, function(p) p[p != ""])
      lengths(parts)
    },
    line = {
      parts = strsplit(x, "\r?\n")
      parts = lapply(parts, function(p) p[p != ""])
      lengths(parts)
    },
    paragraph = {
      parts = strsplit(trimws(x), "\r?\n[[:space:]]*\r?\n")
      parts = lapply(parts, function(p) p[p != ""])
      lengths(parts)
    },
    char = {
      parts = strsplit(gsub("[[:space:]]", "", x), "")
      parts = lapply(parts, function(p) p[p != ""])
      lengths(parts)
    }
  )
}

#' Summarise a token table
#'
#' @param tokens A data frame as returned by [tokenize()].
#' @return A list with elements `nTokens`, `nTypes`, `typeTokenRatio`,
#'   `meanLength`, `longest` and `byType`, a named integer vector of token
#'   counts per rule.
#' @export
tokenSummary = function(tokens) {
  nTokens = nrow(tokens)
  nTypes = length(unique(tokens$token))
  if (nTokens > 0) {
    typeTokenRatio = nTypes / nTokens
    meanLength = mean(nchar(tokens$token))
    longest = tokens$token[which.max(nchar(tokens$token))]
  } else {
    typeTokenRatio = NA_real_
    meanLength = NA_real_
    longest = NA_character_
  }

  byType = table(factor(tokens$type, levels = names(tokenRules())))
  list(
    nTokens = nTokens,
    nTypes = nTypes,
    typeTokenRatio = typeTokenRatio,
    meanLength = meanLength,
    longest = longest,
    byType = c(byType)
  )
}

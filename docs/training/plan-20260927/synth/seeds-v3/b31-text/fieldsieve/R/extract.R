#' Find all matches of a pattern
#'
#' @param x A character vector of documents.
#' @param pattern A PCRE pattern.
#' @param unique Drop repeated matches within a document?
#' @return A list with one character vector of matches per element of `x`.
#'   Missing documents give an empty vector.
#' @keywords internal
extract_matches <- function(x, pattern, unique = TRUE) {
  x[is.na(x)] <- ""
  hits <- regmatches(x, gregexpr(pattern, x, perl = TRUE))
  if (unique) {
    hits <- lapply(hits, unique)
  }
  return(hits)
}

#' Extract entities of one type
#'
#' Phone numbers lose their separators, hashtags lose the leading `#` and
#' URLs lose trailing sentence punctuation.
#'
#' @param x A character vector of documents.
#' @param type Entity type: one of `"email"`, `"url"`, `"phone"`, `"hashtag"`
#'   or `"date"`.
#' @param unique Drop repeated matches within a document?
#' @param lower Convert every match to lower case?
#' @return A list with one character vector of matches per element of `x`.
#' @family extractors
#' @export
extract_entities <- function(x, type = c("email", "url", "phone", "hashtag", "date"),
                             unique = TRUE, lower = FALSE) {
  type <- match.arg(type)
  # Each arm sets `pattern`, then extracts and cleans the matches
  hits <- switch(type,
    email = {
      pattern <- "[[:alnum:]._%+-]+@[[:alnum:].-]+\\.[[:alpha:]]{2,}"
      found <- extract_matches(x, pattern, unique = FALSE)
      found
    },
    url = {
      pattern <- "https?://[^[:space:]<>\"]+"
      found <- extract_matches(x, pattern, unique = FALSE)
      found <- lapply(found, function(v) sub("[.,;:!?)]+$", "", v))
      found
    },
    phone = {
      pattern <- "\\+?[0-9][0-9 ()-]{5,}[0-9]"
      found <- extract_matches(x, pattern, unique = FALSE)
      found <- lapply(found, function(v) gsub("[^0-9+]", "", v))
      found
    },
    hashtag = {
      pattern <- "#[[:alnum:]_]+"
      found <- extract_matches(x, pattern, unique = FALSE)
      found <- lapply(found, function(v) sub("^#", "", v))
      found
    },
    date = {
      pattern <- "\\b[0-9]{4}[-/.][0-9]{2}[-/.][0-9]{2}\\b"
      found <- extract_matches(x, pattern, unique = FALSE)
      found <- lapply(found, function(v) gsub("[/.]", "-", v))
      found
    }
  )

  if (lower) {
    hits <- lapply(hits, tolower)
  }
  if (unique) {
    hits <- lapply(hits, unique)
  }
  return(hits)
}

#' Extract entities into a long table
#'
#' @param x A character vector of documents.
#' @param types The entity types to extract, in output order.
#' @param lower Convert every match to lower case?
#' @return A data frame with columns `doc`, `type` and `value`, one row per
#'   distinct match within a document.
#' @family extractors
#' @export
entity_table <- function(x, types = c("email", "url", "phone", "hashtag", "date"),
                         lower = FALSE) {
  types <- match.arg(types, several.ok = TRUE)
  parts <- vector("list", length(types))
  names(parts) <- types

  for (type in types) {
    hits <- extract_entities(x, type = type, unique = TRUE, lower = lower)
    doc <- rep(seq_along(x), lengths(hits))
    parts[[type]] <- data.frame(
      doc = doc,
      type = rep(type, length(doc)),
      value = unlist(hits, use.names = FALSE),
      stringsAsFactors = FALSE
    )
  }

  out <- do.call(rbind, unname(parts))
  out <- out[order(out$doc), , drop = FALSE]
  rownames(out) <- NULL
  return(out)
}

#' Count entities per document
#'
#' @param x A character vector of documents.
#' @param unique Count repeated matches within a document only once?
#' @return A data frame with one row per element of `x` and the integer
#'   columns `n_email`, `n_url`, `n_phone`, `n_hashtag`, `n_date` and
#'   `n_total`.
#' @family extractors
#' @export
entity_counts <- function(x, unique = FALSE) {
  if (!is.character(x)) {
    stop("`x` must be a character vector.", call. = FALSE)
  }

  # Count `email` matches
  emails <- extract_entities(x, "email", unique = unique)
  n_email <- lengths(emails)

  # Count `url` matches
  urls <- extract_entities(x, "url", unique = unique)
  n_url <- lengths(urls)

  # Count `phone` matches
  phones <- extract_entities(x, "phone", unique = unique)
  n_phone <- lengths(phones)

  # Count `hashtag` matches
  hashtags <- extract_entities(x, "hashtag", unique = unique)
  n_hashtag <- lengths(hashtags)

  # Count `date` matches
  dates <- extract_entities(x, "date", unique = unique)
  n_date <- lengths(dates)

  n_total <- n_email + n_url + n_phone + n_hashtag + n_date
  return(data.frame(
    n_email = n_email,
    n_url = n_url,
    n_phone = n_phone,
    n_hashtag = n_hashtag,
    n_date = n_date,
    n_total = n_total
  ))
}

#' Extract the first entity of each type
#'
#' @param x A single document.
#' @param default Value used when a type has no match.
#' @return A named character vector with elements `email`, `url`, `phone`,
#'   `hashtag` and `date`.
#' @keywords internal
first_entities <- function(x, default = NA_character_) {
  stopifnot(is.character(x), length(x) == 1L)
  out <- c(email = default, url = default, phone = default, hashtag = default,
           date = default)
  if (is.na(x)) {
    return(out)
  }

  emails <- extract_entities(x, "email")[[1]]
  if (length(emails) > 0L) {
    out[["email"]] <- emails[[1]]
  }

  urls <- extract_entities(x, "url")[[1]]
  if (length(urls) > 0L) {
    out[["url"]] <- urls[[1]]
  }

  phones <- extract_entities(x, "phone")[[1]]
  if (length(phones) > 0L) {
    out[["phone"]] <- phones[[1]]
  }

  hashtags <- extract_entities(x, "hashtag")[[1]]
  if (length(hashtags) > 0L) {
    out[["hashtag"]] <- hashtags[[1]]
  }

  dates <- extract_entities(x, "date")[[1]]
  if (length(dates) > 0L) {
    out[["date"]] <- dates[[1]]
  }
  return(out)
}

#' Locate matches of a pattern
#'
#' @param x A character vector of documents.
#' @param pattern A PCRE pattern.
#' @param context Number of characters of context kept on each side.
#' @return A data frame with columns `doc`, `start`, `end`, `match`, `before`
#'   and `after`, one row per match.
#' @export
locate_matches <- function(x, pattern, context = 10L) {
  pos <- gregexpr(pattern, x, perl = TRUE)
  rows <- list()
  for (i in seq_along(x)) {
    starts <- as.integer(pos[[i]])
    if (is.na(starts[1]) || starts[1] < 0L) {
      next
    }
    ends <- starts + attr(pos[[i]], "match.length") - 1L
    n <- nchar(x[[i]])

    # Context before the match
    before_from <- pmax(1L, starts - context)
    before <- substring(x[[i]], before_from, starts - 1L)

    # Context after the match
    after_to <- pmin(n, ends + context)
    after <- substring(x[[i]], ends + 1L, after_to)

    rows[[length(rows) + 1L]] <- data.frame(
      doc = rep(i, length(starts)),
      start = starts,
      end = ends,
      match = substring(x[[i]], starts, ends),
      before = before,
      after = after,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) {
    return(data.frame(doc = integer(), start = integer(), end = integer(),
                      match = character(), before = character(), after = character()))
  }
  return(do.call(rbind, rows))
}

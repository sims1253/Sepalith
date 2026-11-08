#' Regular expressions for redactable entities
#'
#' @return A named list of PCRE patterns with elements `email`, `url`,
#'   `phone` and `hashtag`.
#' @keywords internal
entity_patterns <- function() {
  return(list(
    email = "[[:alnum:]._%+-]+@[[:alnum:].-]+\\.[[:alpha:]]{2,}",
    url = "https?://[^[:space:]<>\"]+[^[:space:]<>\".,;:!?)]",
    phone = "\\+?[0-9][0-9 ()-]{5,}[0-9]",
    hashtag = "#[[:alnum:]_]+"
  ))
}

#' Mask the middle of strings
#'
#' @param x A character vector.
#' @param keep_head Number of leading characters left visible.
#' @param keep_tail Number of trailing characters left visible.
#' @param char The masking character.
#' @return A character vector the same length as `x`. Strings too short to
#'   hide anything are masked completely.
#' @export
mask_keep <- function(x, keep_head = 1L, keep_tail = 1L, char = "*") {
  n <- nchar(x)
  short <- n <= keep_head + keep_tail
  head <- ifelse(short, "", substr(x, 1L, keep_head))
  tail <- ifelse(short, "", substr(x, n - keep_tail + 1L, n))
  hidden <- ifelse(short, n, n - keep_head - keep_tail)
  stars <- vapply(hidden, function(k) strrep(char, k), character(1))
  return(paste0(head, stars, tail))
}

#' Build the replacement for one matched entity
#'
#' @param match The matched text.
#' @param type The entity type of the match.
#' @param style How to replace the match: `"tag"` inserts the upper-case type
#'   in brackets, `"stars"` masks every character, `"ends"` keeps the first
#'   and last character, `"digest"` inserts the type and a short checksum of
#'   the match, and `"drop"` removes the match.
#' @return A single string.
#' @keywords internal
replacement_for <- function(match, type, style = c("tag", "stars", "ends", "digest", "drop")) {
  style <- match.arg(style)
  switch(style,
    tag = {
      label <- toupper(type)
      sprintf("[%s]", label)
    },
    stars = {
      width <- nchar(match)
      strrep("*", width)
    },
    ends = {
      mask_keep(match, keep_head = 1L, keep_tail = 1L)
    },
    digest = {
      # Same input, same checksum: lets readers see repeated values
      codes <- utf8ToInt(match)
      checksum <- sum(codes * seq_along(codes)) %% 65536L
      sprintf("[%s:%04x]", toupper(type), checksum)
    },
    drop = {
      ""
    }
  )
}

#' Redact entities in text
#'
#' Entities are replaced in the order `email`, `url`, `phone`, `hashtag`, so
#' an address inside a URL is redacted as an email.
#'
#' @param x A character vector of documents.
#' @param types The entity types to redact.
#' @param style How to replace each match; see [replacement_for()].
#' @return A character vector the same length as `x`.
#' @export
redact_entities <- function(x, types = c("email", "url", "phone", "hashtag"),
                            style = "tag") {
  types <- match.arg(types, several.ok = TRUE)
  patterns <- entity_patterns()

  # Redact `email`
  if ("email" %in% types) {
    hits <- gregexpr(patterns[["email"]], x, perl = TRUE)
    regmatches(x, hits) <- lapply(regmatches(x, hits), function(m) {
      vapply(m, replacement_for, character(1), type = "email", style = style)
    })
  }

  # Redact `url`
  if ("url" %in% types) {
    hits <- gregexpr(patterns[["url"]], x, perl = TRUE)
    regmatches(x, hits) <- lapply(regmatches(x, hits), function(m) {
      vapply(m, replacement_for, character(1), type = "url", style = style)
    })
  }

  # Redact `phone`
  if ("phone" %in% types) {
    hits <- gregexpr(patterns[["phone"]], x, perl = TRUE)
    regmatches(x, hits) <- lapply(regmatches(x, hits), function(m) {
      vapply(m, replacement_for, character(1), type = "phone", style = style)
    })
  }

  # Redact `hashtag`
  if ("hashtag" %in% types) {
    hits <- gregexpr(patterns[["hashtag"]], x, perl = TRUE)
    regmatches(x, hits) <- lapply(regmatches(x, hits), function(m) {
      vapply(m, replacement_for, character(1), type = "hashtag", style = style)
    })
  }
  return(x)
}

#' Mask an email address
#'
#' The local part, the domain name and the top-level domain are masked
#' separately. Values that are not email addresses become `NA`.
#'
#' @param x A character vector of email addresses.
#' @param keep Number of leading characters left visible in each part.
#' @param keep_domain Leave the domain name unmasked?
#' @param keep_tld Leave the top-level domain unmasked?
#' @return A character vector the same length as `x`.
#' @export
mask_email <- function(x, keep = 1L, keep_domain = FALSE, keep_tld = TRUE) {
  if (!is.numeric(keep) || length(keep) != 1L || keep < 0) {
    stop("`keep` must be a single non-negative number.", call. = FALSE)
  }
  parts <- regmatches(x, regexec("^([^@]+)@(.+)\\.([[:alpha:]]+)$", x))
  out <- rep(NA_character_, length(x))
  for (i in seq_along(parts)) {
    p <- parts[[i]]
    if (length(p) == 0L) {
      next
    }

    # Mask the local part
    local <- mask_keep(p[[2]], keep_head = keep, keep_tail = 0L)

    # Mask the domain part
    if (keep_domain) {
      domain <- p[[3]]
    } else {
      domain <- mask_keep(p[[3]], keep_head = keep, keep_tail = 0L)
    }

    # Mask the top-level domain
    if (keep_tld) {
      tld <- p[[4]]
    } else {
      tld <- mask_keep(p[[4]], keep_head = 0L, keep_tail = 0L)
    }

    out[[i]] <- sprintf("%s@%s.%s", local, domain, tld)
  }
  return(out)
}

#' Summarise the effect of a redaction
#'
#' @param before A character vector of original documents.
#' @param after The redacted documents, in the same order.
#' @param types Entity types whose `"tag"` replacements are counted.
#' @return A list with elements `n_docs`, `n_changed`, `share_changed`,
#'   `chars_removed` and `tags`, a named integer vector of tag counts per
#'   type.
#' @export
redaction_summary <- function(before, after, types = names(entity_patterns())) {
  stopifnot(length(before) == length(after))
  n_docs <- length(before)
  changed <- before != after
  n_changed <- sum(changed, na.rm = TRUE)
  share_changed <- if (n_docs > 0L) n_changed / n_docs else NA_real_
  chars_removed <- sum(pmax(nchar(before) - nchar(after), 0L), na.rm = TRUE)

  # Count the tags left behind by style = "tag"
  tags <- integer(length(types))
  names(tags) <- types
  for (type in types) {
    tag <- sprintf("[%s]", toupper(type))
    hits <- gregexpr(tag, after, fixed = TRUE)
    tags[[type]] <- sum(vapply(hits, function(h) sum(h > 0L), integer(1)))
  }

  return(list(
    n_docs = n_docs,
    n_changed = n_changed,
    share_changed = share_changed,
    chars_removed = chars_removed,
    tags = tags
  ))
}

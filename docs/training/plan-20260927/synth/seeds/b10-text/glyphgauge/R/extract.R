#' Regular expressions used by the extractors
#'
#' @return A named list of PCRE patterns.
#' @noRd
extractPatterns = function() {
  list(
    email = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
    url = "https?://[^\\s<>\"')]+",
    isoDate = "\\b\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])\\b",
    dmyDate = "\\b(0?[1-9]|[12]\\d|3[01])\\.(0?[1-9]|1[0-2])\\.\\d{4}\\b",
    ticketId = "\\b[A-Z]{2,5}-\\d{1,6}\\b"
  )
}

#' Extract all matches of a pattern from each string
#'
#' Thin wrapper around [base::gregexpr()] and [base::regmatches()] that
#' keeps `NA` inputs as empty results and optionally deduplicates.
#'
#' @param text A character vector to search.
#' @param pattern A Perl-compatible regular expression.
#' @param unique Logical; drop repeated matches within each string?
#' @return A list of character vectors, one element per string in `text`.
#' @keywords internal
extractPattern = function(text, pattern, unique = TRUE) {
  text[is.na(text)] = ""
  hits = regmatches(text, gregexpr(pattern, text, perl = TRUE))
  if (unique) {
    hits = lapply(hits, base::unique)
  }
  names(hits) = names(text)
  hits
}

#' Extract e-mail addresses
#'
#' @param text A character vector to search.
#' @param unique Logical; drop repeated matches within each string?
#' @return A list of character vectors, one element per string in `text`.
#' @family extractors
#' @export
extractEmails = function(text, unique = TRUE) {
  hits = extractPattern(text, extractPatterns()$email, unique = unique)
  lapply(hits, tolower)
}

#' Extract web addresses
#'
#' Trailing punctuation such as a full stop at the end of a sentence is
#' trimmed from each match.
#'
#' @param text A character vector to search.
#' @param unique Logical; drop repeated matches within each string?
#' @return A list of character vectors, one element per string in `text`.
#' @family extractors
#' @export
extractUrls = function(text, unique = TRUE) {
  hits = extractPattern(text, extractPatterns()$url, unique = FALSE)
  hits = lapply(hits, function(h) sub("[.,;:!?]+$", "", h))
  if (unique) {
    hits = lapply(hits, base::unique)
  }
  hits
}

#' Extract ticket identifiers
#'
#' Matches identifiers of the form `ABC-123`, optionally restricted to a
#' set of project prefixes.
#'
#' @param text A character vector to search.
#' @param unique Logical; drop repeated matches within each string?
#' @param prefixes Optional character vector of allowed project prefixes.
#' @return A list of character vectors, one element per string in `text`.
#' @family extractors
#' @export
extractTicketIds = function(text, unique = TRUE, prefixes = NULL) {
  hits = extractPattern(text, extractPatterns()$ticketId, unique = unique)
  if (is.null(prefixes)) {
    return(hits)
  }
  lapply(hits, function(h) h[sub("-.*$", "", h) %in% prefixes])
}

#' Extract and parse calendar dates
#'
#' Finds ISO (`2024-03-01`) and dotted day-month-year (`1.3.2024`) dates
#' and converts both to `Date`. Strings that look like dates but are not
#' valid calendar days are dropped.
#'
#' @param text A character vector to search.
#' @param unique Logical; drop repeated matches within each string?
#' @return A list of `Date` vectors, one element per string in `text`.
#' @family extractors
#' @export
extractDates = function(text, unique = TRUE) {
  pats = extractPatterns()
  iso = extractPattern(text, pats$isoDate, unique = FALSE)
  dmy = extractPattern(text, pats$dmyDate, unique = FALSE)
  out = vector("list", length(text))
  for (i in seq_along(text)) {
    d1 = as.Date(iso[[i]], format = "%Y-%m-%d")
    d2 = as.Date(dmy[[i]], format = "%d.%m.%Y")
    dates = c(d1, d2)
    dates = dates[!is.na(dates)]
    if (unique) {
      dates = base::unique(dates)
    }
    out[[i]] = sort(dates)
  }
  names(out) = names(text)
  out
}

#' Extract matches with stringr
#'
#' A tidyverse-flavoured alternative that returns a long data frame
#' instead of a list, which is handier inside pipelines.
#'
#' @param text A character vector to search.
#' @param what Which kind of entity to extract.
#' @return A data frame with columns `row`, `kind` and `match`.
#' @family extractors
#' @export
extractTable = function(text, what = c("email", "url", "ticketId", "isoDate")) {
  what = match.arg(what)
  pattern = stringr::regex(extractPatterns()[[what]])
  hits = stringr::str_extract_all(text, pattern)
  lens = lengths(hits)
  data.frame(
    row = rep(seq_along(text), lens),
    kind = rep(what, sum(lens)),
    match = unlist(hits, use.names = FALSE),
    stringsAsFactors = FALSE
  )
}

#' Mask e-mail addresses for display
#'
#' Keeps the first character of the local part and the full domain,
#' replacing the rest with asterisks.
#'
#' @param text A character vector to search.
#' @return `text` with every e-mail address masked.
#' @export
maskEmails = function(text) {
  stringr::str_replace_all(text, extractPatterns()$email, function(m) {
    local = stringr::str_extract(m, "^[^@]+")
    domain = stringr::str_remove(m, "^[^@]+")
    paste0(
      stringr::str_sub(local, 1, 1),
      strrep("*", pmax(nchar(local) - 1L, 0L)),
      domain
    )
  })
}

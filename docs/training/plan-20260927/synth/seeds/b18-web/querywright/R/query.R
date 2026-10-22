#' Percent-encode a query component
#'
#' Encodes every character outside the unreserved set, so the result is
#' safe as either a key or a value in a query string.
#'
#' @param x Character vector to encode.
#' @param space_as_plus If `TRUE`, spaces become `+` instead of `%20`.
#' @return A character vector of the same length.
#' @keywords internal
encode_component <- function(x, space_as_plus = FALSE) {
  out <- vapply(x, utils::URLencode, character(1), reserved = TRUE, USE.NAMES = FALSE)
  if (space_as_plus) {
    out <- gsub("%20", "+", out, fixed = TRUE)
  }
  return(out)
}

#' Decode a percent-encoded query component
#'
#' Turns `+` into spaces before decoding, as form-encoded query strings
#' require.
#'
#' @param x Character vector to decode.
#' @return A character vector of the same length.
#' @keywords internal
decode_component <- function(x) {
  x <- gsub("+", " ", x, fixed = TRUE)
  out <- vapply(x, utils::URLdecode, character(1), USE.NAMES = FALSE)
  return(out)
}

#' Build a query string from a list of parameters
#'
#' `NULL` values are dropped, logicals become `true`/`false`, dates are
#' formatted as ISO 8601 and vectors are repeated as `key=a&key=b`.
#'
#' @param params A named list of query parameters.
#' @param space_as_plus If `TRUE`, spaces become `+` instead of `%20`.
#' @return A single query string without the leading `?`.
#' @family query helpers
#' @export
query_build <- function(params, space_as_plus = FALSE) {
  if (length(params) == 0) {
    return("")
  }
  if (is.null(names(params)) || any(!nzchar(names(params)))) {
    stop("All query parameters must be named.", call. = FALSE)
  }
  pairs <- character()
  for (key in names(params)) {
    value <- params[[key]]
    if (is.null(value)) {
      next
    }
    if (is.logical(value)) {
      value <- ifelse(value, "true", "false")
    } else if (inherits(value, "Date")) {
      value <- format(value, "%Y-%m-%d")
    } else if (inherits(value, "POSIXt")) {
      value <- format(value, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    } else {
      value <- as.character(value)
    }
    value <- value[!is.na(value)]
    encoded <- paste0(
      encode_component(key, space_as_plus),
      "=",
      encode_component(value, space_as_plus)
    )
    pairs <- c(pairs, encoded[seq_along(value)])
  }
  return(paste(pairs, collapse = "&"))
}

#' Parse a query string into a list of parameters
#'
#' Repeated keys are collected into character vectors and keys without a
#' value map to `""`.
#'
#' @param query A single query string, with or without the leading `?`.
#' @return A named list of character vectors.
#' @family query helpers
#' @export
query_parse <- function(query) {
  query <- sub("^\\?", "", query)
  if (!nzchar(query)) {
    return(list())
  }
  pairs <- strsplit(query, "&", fixed = TRUE)[[1]]
  pairs <- pairs[nzchar(pairs)]
  params <- list()
  for (pair in pairs) {
    eq <- regexpr("=", pair, fixed = TRUE)
    if (eq > 0) {
      key <- decode_component(substr(pair, 1, eq - 1))
      value <- decode_component(substr(pair, eq + 1, nchar(pair)))
    } else {
      key <- decode_component(pair)
      value <- ""
    }
    params[[key]] <- c(params[[key]], value)
  }
  return(params)
}

#' Add or replace query parameters on a URL
#'
#' Existing parameters with the same name are replaced; others are kept in
#' their original order.
#'
#' @param url A single URL string.
#' @param ... Named query parameters; `NULL` removes a parameter.
#' @return The modified URL.
#' @family query modifiers
#' @export
url_modify_query <- function(url, ...) {
  updates <- list(...)
  parts <- url_parts(url)
  params <- query_parse(parts$query)
  for (key in names(updates)) {
    params[key] <- list(updates[[key]])
  }
  params <- params[!vapply(params, is.null, logical(1))]
  parts$query <- query_build(params)
  return(url_assemble(parts))
}

#' Remove query parameters from a URL
#'
#' Drops the named parameters and keeps all others, which is useful for
#' stripping tracking parameters before deduplicating URLs.
#'
#' @param url A single URL string.
#' @param keys Character vector of parameter names; entries ending in `*`
#'   match by prefix.
#' @return The modified URL.
#' @family query modifiers
#' @export
url_drop_query <- function(url, keys) {
  parts <- url_parts(url)
  params <- query_parse(parts$query)
  prefixes <- sub("\\*$", "", keys[endsWith(keys, "*")])
  exact <- keys[!endsWith(keys, "*")]
  drop <- names(params) %in% exact |
    vapply(names(params), function(k) any(startsWith(k, prefixes)), logical(1))
  parts$query <- query_build(params[!drop])
  return(url_assemble(parts))
}

#' Keep only selected query parameters on a URL
#'
#' The counterpart of [url_drop_query()]: every parameter not listed in
#' `keys` is removed.
#'
#' @param url A single URL string.
#' @param keys Character vector of parameter names; entries ending in `*`
#'   match by prefix.
#' @return The modified URL.
#' @family query modifiers
#' @export
url_keep_query <- function(url, keys) {
  parts <- url_parts(url)
  params <- query_parse(parts$query)
  prefixes <- sub("\\*$", "", keys[endsWith(keys, "*")])
  exact <- keys[!endsWith(keys, "*")]
  keep <- names(params) %in% exact |
    vapply(names(params), function(k) any(startsWith(k, prefixes)), logical(1))
  parts$query <- query_build(params[keep])
  return(url_assemble(parts))
}

#' Sort the query parameters of a URL
#'
#' Orders parameters by name so that equivalent URLs compare equal.
#'
#' @param url A single URL string.
#' @return The modified URL.
#' @family query modifiers
#' @export
url_sort_query <- function(url) {
  parts <- url_parts(url)
  params <- query_parse(parts$query)
  if (length(params) < 2) {
    return(url)
  }
  parts$query <- query_build(params[order(names(params))])
  return(url_assemble(parts))
}

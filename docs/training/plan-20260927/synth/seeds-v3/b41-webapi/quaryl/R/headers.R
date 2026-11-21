#' Build the request headers
#'
#' One header per argument; `NULL` arguments are left out.
#'
#' @param token Bearer token, sent as `Authorization`.
#' @param accept Media type sent as `Accept`.
#' @param user_agent Client identifier sent as `User-Agent`.
#' @param content_type Media type of the body, sent as `Content-Type`.
#' @param request_id Correlation id sent as `X-Request-Id`.
#' @param if_none_match Cached ETag sent as `If-None-Match`.
#' @return A named character vector of headers.
#' @export
build_request_headers <- function(token = NULL, accept = "application/json",
                                  user_agent = "quaryl", content_type = NULL,
                                  request_id = NULL, if_none_match = NULL) {
  headers <- character(0)
  # Header `Authorization`
  if (!is.null(token)) {
    if (!is.character(token) || length(token) != 1L || !nzchar(token)) {
      stop(sprintf("`%s` must be a single non-empty string.", "token"), call. = FALSE)
    }
    headers[["Authorization"]] <- paste("Bearer", token)
  }

  # Header `Accept`
  if (!is.null(accept)) {
    if (!is.character(accept) || length(accept) != 1L || !nzchar(accept)) {
      stop(sprintf("`%s` must be a single non-empty string.", "accept"), call. = FALSE)
    }
    headers[["Accept"]] <- accept
  }

  # Header `User-Agent`
  if (!is.null(user_agent)) {
    if (!is.character(user_agent) || length(user_agent) != 1L || !nzchar(user_agent)) {
      stop(sprintf("`%s` must be a single non-empty string.", "user_agent"), call. = FALSE)
    }
    headers[["User-Agent"]] <- user_agent
  }

  # Header `Content-Type`
  if (!is.null(content_type)) {
    if (!is.character(content_type) || length(content_type) != 1L || !nzchar(content_type)) {
      stop(sprintf("`%s` must be a single non-empty string.", "content_type"), call. = FALSE)
    }
    headers[["Content-Type"]] <- content_type
  }

  # Header `X-Request-Id`
  if (!is.null(request_id)) {
    if (!is.character(request_id) || length(request_id) != 1L || !nzchar(request_id)) {
      stop(sprintf("`%s` must be a single non-empty string.", "request_id"), call. = FALSE)
    }
    headers[["X-Request-Id"]] <- request_id
  }

  # Header `If-None-Match`
  if (!is.null(if_none_match)) {
    if (!is.character(if_none_match) || length(if_none_match) != 1L || !nzchar(if_none_match)) {
      stop(sprintf("`%s` must be a single non-empty string.", "if_none_match"), call. = FALSE)
    }
    headers[["If-None-Match"]] <- if_none_match
  }

  return(headers)
}

#' Encode raw bytes as base64
#'
#' Base R has no exported base64 encoder, so this works on groups of three bytes.
#'
#' @param x A raw vector.
#' @return A single string.
#' @keywords internal
base64_encode <- function(x) {
  alphabet <- c(LETTERS, letters, 0:9, "+", "/")
  n <- length(x)
  if (n == 0L) {
    return("")
  }
  pad <- (3L - n %% 3L) %% 3L
  bytes <- c(as.integer(x), rep(0L, pad))
  groups <- matrix(bytes, nrow = 3L)
  out <- character(ncol(groups) * 4L)
  for (i in seq_len(ncol(groups))) {
    b <- groups[, i]
    v <- b[1] * 65536L + b[2] * 256L + b[3]
    idx <- c(v %/% 262144L, (v %/% 4096L) %% 64L, (v %/% 64L) %% 64L, v %% 64L)
    out[(i - 1L) * 4L + 1:4] <- alphabet[idx + 1L]
  }
  if (pad > 0L) {
    out[(length(out) - pad + 1L):length(out)] <- "="
  }

  return(paste(out, collapse = ""))
}

#' Build an authentication header
#'
#' Supports the authentication schemes used by the services we wrap.
#'
#' @param credentials A named list; which elements are used depends on
#'   `scheme`.
#' @param scheme One of `"bearer"` (`token`), `"basic"` (`user`,
#'   `password`), `"token"` (`token`), `"apikey"` (`key`, `header`) or
#'   `"none"`.
#' @return A named character vector with zero or one header.
#' @export
auth_header <- function(credentials = list(),
                        scheme = c("bearer", "basic", "token", "apikey", "none")) {
  scheme <- match.arg(scheme)
  need <- function(field) {
    value <- credentials[[field]]
    if (!is.character(value) || length(value) != 1L || !nzchar(value)) {
      stop(sprintf("Scheme `%s` needs `credentials$%s`.", scheme, field), call. = FALSE)
    }
    value
  }

  header <- switch(scheme,
    bearer = {
      token <- need("token")
      c(Authorization = paste("Bearer", token))
    },
    basic = {
      user <- need("user")
      password <- need("password")
      raw_pair <- charToRaw(paste0(user, ":", password))
      c(Authorization = paste("Basic", base64_encode(raw_pair)))
    },
    token = {
      token <- need("token")
      c(Authorization = paste("token", token))
    },
    apikey = {
      key <- need("key")
      header_name <- if (is.null(credentials$header)) "X-Api-Key" else credentials$header
      stats::setNames(key, header_name)
    },
    none = {
      character(0)
    }
  )
  return(header)
}

#' Parse raw response header lines
#'
#' Header names are matched case-insensitively; the last occurrence wins.
#'
#' @param lines Character vector of `Name: value` lines.
#' @return A list with elements `content_type`, `content_length`, `etag`,
#'   `last_modified`, `retry_after` and `link`; absent headers are `NA`.
#' @export
parse_response_headers <- function(lines) {
  lines <- lines[grepl(":", lines, fixed = TRUE)]
  keys <- tolower(trimws(sub(":.*$", "", lines)))
  values <- trimws(sub("^[^:]*:", "", lines))
  lookup <- function(name) {
    hit <- which(keys == name)
    if (length(hit) == 0L) NA_character_ else values[[max(hit)]]
  }

  out <- list()
  # Field `content_type`: media type without parameters
  value <- lookup("content-type")
  out$content_type <- if (is.na(value)) NA_character_ else trimws(sub(";.*$", "", value))
  # Field `content_length`: integer byte count
  value <- lookup("content-length")
  out$content_length <- if (is.na(value)) NA_integer_ else as.integer(value)
  # Field `etag`: strip the weak prefix and quotes
  value <- lookup("etag")
  out$etag <- if (is.na(value)) NA_character_ else gsub("^W/|\"", "", value)
  # Field `last_modified`: HTTP date as POSIXct
  value <- lookup("last-modified")
  out$last_modified <- if (is.na(value)) {
    as.POSIXct(NA)
  } else {
    as.POSIXct(value, format = "%a, %d %b %Y %H:%M:%S", tz = "GMT")
  }
  # Field `retry_after`: seconds to wait
  value <- lookup("retry-after")
  out$retry_after <- if (is.na(value)) NA_real_ else as.numeric(value)
  # Field `link`: raw Link header
  value <- lookup("link")
  out$link <- if (is.na(value)) NA_character_ else value
  return(out)
}

#' Extract page links from a Link header
#'
#' Reads the `rel="next"`, `rel="prev"`, `rel="first"` and `rel="last"`
#' entries of an RFC 8288 Link header.
#'
#' @param link A single Link header value, or `NA`.
#' @return A list with elements `next`, `prev`, `first` and `last`, each a
#'   URL or `NA`.
#' @export
parse_link_header <- function(link) {
  out <- list(`next` = NA_character_, prev = NA_character_, first = NA_character_,
    last = NA_character_)
  if (is.na(link) || !nzchar(link)) {
    return(out)
  }
  entries <- trimws(strsplit(link, ",", fixed = TRUE)[[1]])
  urls <- sub("^<([^>]*)>.*$", "\\1", entries)
  rels <- sub('^.*rel="?([a-z]+)"?.*$', "\\1", entries)
  # Relation `next`
  hit <- which(rels == "next")
  if (length(hit) > 0L) {
    out[["next"]] <- urls[[hit[1]]]
  }

  # Relation `prev`
  hit <- which(rels == "prev")
  if (length(hit) > 0L) {
    out[["prev"]] <- urls[[hit[1]]]
  }

  # Relation `first`
  hit <- which(rels == "first")
  if (length(hit) > 0L) {
    out[["first"]] <- urls[[hit[1]]]
  }

  # Relation `last`
  hit <- which(rels == "last")
  if (length(hit) > 0L) {
    out[["last"]] <- urls[[hit[1]]]
  }

  return(out)
}

#' Check that a response has the expected content type
#'
#' Each expected format accepts a small set of media types.
#'
#' @param content_type The parsed `Content-Type`, as returned in
#'   `parse_response_headers()$content_type`.
#' @param expect One of `"json"`, `"xml"`, `"csv"`, `"text"` or `"binary"`.
#' @return `content_type`, invisibly; an error when it does not match.
#' @export
check_content_type <- function(content_type,
                               expect = c("json", "xml", "csv", "text", "binary")) {
  expect <- match.arg(expect)
  if (is.na(content_type)) {
    stop("Response has no Content-Type header.", call. = FALSE)
  }
  type <- tolower(content_type)

  ok <- switch(expect,
    json = {
      allowed <- c("application/json", "text/json")
      type %in% allowed || grepl("\\+json$", type)
    },
    xml = {
      allowed <- c("application/xml", "text/xml")
      type %in% allowed || grepl("\\+xml$", type)
    },
    csv = {
      allowed <- c("text/csv", "application/csv")
      type %in% allowed
    },
    text = {
      allowed <- c("text/plain")
      type %in% allowed || startsWith(type, "text/")
    },
    binary = {
      allowed <- c("application/octet-stream")
      type %in% allowed || grepl("^(image|audio|video)/", type)
    }
  )
  if (!ok) {
    stop(sprintf("Expected a %s response but got `%s`.", expect, content_type),
      call. = FALSE)
  }

  return(invisible(content_type))
}

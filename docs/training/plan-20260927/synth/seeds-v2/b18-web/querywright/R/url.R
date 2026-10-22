#' @title Split a URL into its components
#' @description Parses an absolute or relative URL with a single regular expression and
#'   returns every component as a string, using `""` for missing parts.
#' @param url A single URL string.
#' @return A list with `scheme`, `host`, `port`, `path`, `query` and
#'   `fragment`.
#' @export
url_parts <- function(url) {
  stopifnot(is.character(url), length(url) == 1)

  # scheme, then //host:port, then path, ?query and #fragment; every part
  # is optional so relative references parse too
  pattern <- paste0(
    "^(?:([A-Za-z][A-Za-z0-9+.-]*):)?",
    "(?://([^/:?#]*)(?::([0-9]+))?)?",
    "([^?#]*)(?:\\?([^#]*))?(?:#(.*))?$"
  )
  m <- regmatches(url, regexec(pattern, url, perl = TRUE))[[1]]
  if (length(m) == 0) {
    stop("Could not parse URL: ", url, call. = FALSE)
  }

  return(list(
    scheme = tolower(m[2]),
    host = tolower(m[3]),
    port = m[4],
    path = m[5],
    query = m[6],
    fragment = m[7]
  ))
}

#' @title Assemble a URL from its components
#' @description The inverse of [url_parts()]: empty components are left out of the
#'   result.
#' @param parts A list as returned by [url_parts()].
#' @return A single URL string.
#' @export
url_assemble <- function(parts) {
  out <- ""
  if (nzchar(parts$scheme)) {
    out <- paste0(parts$scheme, ":")
  }
  if (nzchar(parts$host)) {
    out <- paste0(out, "//", parts$host)
    if (nzchar(parts$port)) {
      out <- paste0(out, ":", parts$port)
    }
  }

  out <- paste0(out, parts$path)
  if (nzchar(parts$query)) {
    out <- paste0(out, "?", parts$query)
  }
  if (nzchar(parts$fragment)) {
    out <- paste0(out, "#", parts$fragment)
  }
  return(out)
}

#' @title Replace the path of a URL
#' @description Parses the URL, swaps in the new path and assembles it again.
#' @param url A single URL string.
#' @param path The new path; a leading slash is added if missing.
#' @return The modified URL.
#' @export
url_set_path <- function(url, path) {
  parts <- url_parts(url)
  if (!startsWith(path, "/")) {
    path <- paste0("/", path)
  }
  parts$path <- path
  return(url_assemble(parts))
}

#' @title Replace the fragment of a URL
#' @description Parses the URL, swaps in the new fragment and assembles it again.
#' @param url A single URL string.
#' @param fragment The new fragment without `#`, or `""` to remove it.
#' @return The modified URL.
#' @export
url_set_fragment <- function(url, fragment) {
  parts <- url_parts(url)
  if (startsWith(fragment, "#")) {
    fragment <- substring(fragment, 2)
  }
  parts$fragment <- fragment
  return(url_assemble(parts))
}

#' @title Replace the scheme of a URL
#' @description Parses the URL, swaps in the new scheme and assembles it again.
#' @param url A single URL string.
#' @param scheme The new scheme, such as `"https"`.
#' @return The modified URL.
#' @export
url_set_scheme <- function(url, scheme) {
  parts <- url_parts(url)
  if (!grepl("^[A-Za-z][A-Za-z0-9+.-]*$", scheme)) {
    stop("Invalid scheme: ", scheme, call. = FALSE)
  }
  parts$scheme <- tolower(scheme)
  return(url_assemble(parts))
}

#' @title Join path segments onto a base URL
#' @description Appends segments to the path of `base`, encoding each one and avoiding
#'   doubled or missing slashes. The query and fragment of `base` are kept.
#' @param base A single base URL.
#' @param ... Path segments, each a single string or number.
#' @return The joined URL.
#' @export
url_join <- function(base, ...) {
  segments <- vapply(list(...), as.character, character(1))
  parts <- url_parts(base)

  # Trim slashes on both sides so exactly one separates each segment.
  path <- sub("/+$", "", parts$path)
  for (segment in segments) {
    segment <- gsub("^/+|/+$", "", segment)
    if (!nzchar(segment)) {
      next
    }
    path <- paste0(path, "/", utils::URLencode(segment, reserved = TRUE))
  }

  parts$path <- if (nzchar(path)) path else "/"
  return(url_assemble(parts))
}

#' @title Normalize a URL for comparison
#' @description Lower-cases scheme and host, drops default ports and fragments, resolves
#'   `.` and `..` path segments and removes a trailing slash.
#' @param url A single URL string.
#' @param keep_fragment If `TRUE`, the fragment is preserved.
#' @return The normalized URL.
#' @export
url_normalize <- function(url, keep_fragment = FALSE) {
  parts <- url_parts(url)
  default_port <- switch(parts$scheme,
    http = "80",
    https = "443",
    ftp = "21",
    ""
  )
  if (identical(parts$port, default_port)) {
    parts$port <- ""
  }

  # Resolve dot segments with a stack; ".." above the root is ignored.
  segments <- strsplit(parts$path, "/", fixed = TRUE)[[1]]
  stack <- character()
  for (segment in segments) {
    if (segment == "" || segment == ".") {
      next
    } else if (segment == "..") {
      stack <- stack[-length(stack)]
    } else {
      stack <- c(stack, segment)
    }
  }
  parts$path <- paste0("/", paste(stack, collapse = "/"))

  if (!keep_fragment) {
    parts$fragment <- ""
  }
  return(url_assemble(parts))
}

#' Create a request throttle
#'
#' Builds a small mutable object that remembers when the last request was
#' sent, so consecutive requests to one host respect a minimum delay.
#'
#' @param delay Minimum number of seconds between two requests.
#' @param jitter Maximum random extra delay in seconds added to each wait.
#' @return An environment of class `pagesieve_throttle`. `throttle_wait()`
#'   returns the number of seconds slept and `throttle_reset()` the throttle,
#'   both invisibly.
#' @export
new_throttle <- function(delay = 2, jitter = 0.5) {
  stopifnot(is.numeric(delay), length(delay) == 1, delay >= 0)
  stopifnot(is.numeric(jitter), length(jitter) == 1, jitter >= 0)

  # An environment, so waits update the shared state in place.
  throttle <- new.env(parent = emptyenv())
  throttle$delay <- delay
  throttle$jitter <- jitter
  throttle$last <- -Inf
  throttle$count <- 0L
  class(throttle) <- "pagesieve_throttle"
  throttle
}

#' @describeIn new_throttle Sleeps for whatever remains of the delay since
#'   the previous request and records the current time as the new reference
#'   point.
#' @param throttle A throttle created by [new_throttle()].
#' @export
throttle_wait <- function(throttle) {
  if (!inherits(throttle, "pagesieve_throttle")) {
    stop("`throttle` must be created with new_throttle().", call. = FALSE)
  }

  now <- as.numeric(Sys.time())
  extra <- stats::runif(1, min = 0, max = throttle$jitter)
  pause <- max(0, throttle$last + throttle$delay + extra - now)
  if (pause > 0) {
    Sys.sleep(pause)
  }

  throttle$last <- as.numeric(Sys.time())
  throttle$count <- throttle$count + 1L
  invisible(pause)
}

#' @describeIn new_throttle Forgets the time of the last request so that
#'   the next call to `throttle_wait()` returns immediately.
#' @export
throttle_reset <- function(throttle) {
  if (!inherits(throttle, "pagesieve_throttle")) {
    stop("`throttle` must be created with new_throttle().", call. = FALSE)
  }

  throttle$last <- -Inf
  throttle$count <- 0L
  invisible(throttle)
}

#' Parse the disallow rules of a robots.txt file
#'
#' Extracts the `Disallow` paths that apply to a given user agent, falling
#' back to the wildcard group when no group names the agent explicitly.
#'
#' @param robots_txt Character vector of robots.txt lines.
#' @param agent User agent string sent with every request.
#' @return A character vector of disallowed path prefixes.
#' @keywords internal
parse_robots_rules <- function(robots_txt, agent = "pagesieve") {
  lines <- trimws(sub("#.*$", "", robots_txt))
  lines <- lines[nzchar(lines)]

  rules <- list()
  current <- character()
  in_agents <- FALSE
  for (line in lines) {
    key <- tolower(trimws(sub(":.*$", "", line)))
    value <- trimws(sub("^[^:]*:", "", line))
    if (key == "user-agent") {
      # consecutive user-agent lines share one rule group
      if (!in_agents) {
        current <- character()
      }
      current <- c(current, tolower(value))
      in_agents <- TRUE
    } else {
      in_agents <- FALSE
      if (key == "disallow" && nzchar(value)) {
        for (ua in current) {
          rules[[ua]] <- c(rules[[ua]], value)
        }
      }
    }
  }

  # An explicit group for our agent replaces the wildcard rules entirely.
  agent <- tolower(agent)
  if (!is.null(rules[[agent]])) {
    return(unique(rules[[agent]]))
  }
  unique(rules[["*"]] %||% character())
}

#' Check whether a URL may be fetched
#'
#' Compares the path component of a URL against a set of disallowed
#' prefixes as returned by [parse_robots_rules()].
#'
#' @param url A single absolute URL.
#' @param rules Character vector of disallowed path prefixes.
#' @return `TRUE` when no rule blocks the URL, `FALSE` otherwise.
#' @keywords internal
path_allowed <- function(url, rules) {
  # Drop scheme and host, keeping path and query.
  path <- sub("^[a-z][a-z0-9+.-]*://[^/]+", "", url, ignore.case = TRUE)
  if (!nzchar(path)) {
    path <- "/"
  }
  if (length(rules) == 0) {
    return(TRUE)
  }

  blocked <- vapply(rules, function(rule) startsWith(path, rule), logical(1))
  !any(blocked)
}

#' Fetch and parse a single page politely
#'
#' Waits on the throttle, performs the request, and retries with a growing
#' back-off when the server answers with 429 or a 5xx status.
#'
#' @param url A single absolute URL.
#' @param throttle A throttle created by [new_throttle()].
#' @param agent User agent string sent with every request.
#' @param retries Maximum number of attempts.
#' @param timeout Request timeout in seconds.
#' @return A list with elements `url`, `status`, `html`, `attempts` and
#'   `error`.
#' @export
fetch_page <- function(url, throttle, agent = "pagesieve", retries = 3, timeout = 20) {
  stopifnot(is.character(url), length(url) == 1, retries >= 1)
  attempt <- 0L
  last_error <- NULL
  result <- list(status = NA_integer_, body = NULL, error = NULL)

  while (attempt < retries) {
    attempt <- attempt + 1L
    throttle_wait(throttle)
    result <- tryCatch(
      {
        resp <- httr2::request(url) |>
          httr2::req_user_agent(agent) |>
          httr2::req_timeout(timeout) |>
          httr2::req_error(is_error = function(resp) FALSE) |>
          httr2::req_perform()
        list(
          status = httr2::resp_status(resp),
          body = httr2::resp_body_string(resp),
          error = NULL
        )
      },
      error = function(e) {
        list(status = NA_integer_, body = NULL, error = conditionMessage(e))
      }
    )

    # Network errors, rate limits and server errors are worth another try;
    # any other status is final.
    retryable <- is.na(result$status) || result$status == 429L || result$status >= 500L
    if (!retryable) {
      break
    }
    last_error <- result$error %||% sprintf("HTTP status %d", result$status)
    # push the next allowed slot further out on every failed attempt
    throttle$last <- throttle$last + throttle$delay * attempt
  }

  ok <- !is.na(result$status) && result$status < 400L
  if (!ok && !is.na(result$status)) {
    last_error <- sprintf("HTTP status %d", result$status)
  }

  list(
    url = url,
    status = result$status,
    html = if (ok) rvest::read_html(result$body) else NULL,
    attempts = attempt,
    error = if (ok) NULL else last_error
  )
}

#' Fetch several pages politely
#'
#' Fetches each URL in turn through one shared throttle, skipping URLs that
#' are blocked by robots.txt rules.
#'
#' @param urls Character vector of absolute URLs.
#' @param throttle A throttle created by [new_throttle()].
#' @param agent User agent string sent with every request.
#' @param rules Character vector of disallowed path prefixes.
#' @return A named list of page results as returned by [fetch_page()].
#' @export
fetch_pages <- function(urls, throttle = new_throttle(), agent = "pagesieve",
                        rules = character()) {
  pages <- vector("list", length(urls))
  names(pages) <- urls

  for (i in seq_along(urls)) {
    if (!path_allowed(urls[[i]], rules)) {
      pages[[i]] <- list(
        url = urls[[i]],
        status = NA_integer_,
        html = NULL,
        attempts = 0L,
        error = "disallowed by robots.txt"
      )
      next
    }

    pages[[i]] <- fetch_page(urls[[i]], throttle, agent = agent)
  }
  pages
}

#' Read a cached page from disk
#'
#' Looks up the raw HTML stored for a URL in a cache directory and parses
#' it when present and not older than `max_age`.
#'
#' @param url A single absolute URL.
#' @param dir Path to the cache directory.
#' @param max_age Maximum age of a cache entry in hours.
#' @return An `xml_document`, or `NULL` when there is no usable entry.
#' @keywords internal
cache_read <- function(url, dir, max_age = 24) {
  # Must match the file naming in cache_write().
  key <- gsub("[^A-Za-z0-9]+", "_", sub("^https?://", "", url))
  path <- file.path(dir, paste0(key, ".html"))
  if (!file.exists(path)) {
    return(NULL)
  }

  age <- difftime(Sys.time(), file.mtime(path), units = "hours")
  if (as.numeric(age) > max_age) {
    return(NULL)
  }
  rvest::read_html(path)
}

#' Write a page to the disk cache
#'
#' Stores the raw HTML of a fetched page under a file name derived from its
#' URL, creating the cache directory if needed.
#'
#' @param url A single absolute URL.
#' @param dir Path to the cache directory.
#' @param html An `xml_document` holding the parsed page.
#' @return The path of the written file, invisibly.
#' @keywords internal
cache_write <- function(url, dir, html) {
  key <- gsub("[^A-Za-z0-9]+", "_", sub("^https?://", "", url))
  path <- file.path(dir, paste0(key, ".html"))

  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  xml2::write_html(html, path)
  invisible(path)
}

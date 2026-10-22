#' Build a base request for the inventory API
#'
#' Sets the base URL, user agent, authentication header and a default timeout.
#' All other request helpers start from this object.
#'
#' @param config A configuration list as returned by [clerk_config()].
#' @return An `httr2_request` object.
#' @export
clerk_request <- function(config = clerk_config()) {
  token <- config$token
  if (is.null(token) || !nzchar(token)) {
    stop(
      "No API token configured; set APICLERK_TOKEN or add `token` to the config.",
      call. = FALSE
    )
  }
  httr2::request(config$base_url) |>
    httr2::req_user_agent(paste0("apiclerk/", config$version, " (R)")) |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_headers(Accept = "application/json") |>
    httr2::req_timeout(config$timeout)
}

#' Add a retry policy to a request
#'
#' Retries on 429 and 5xx responses as well as transient network failures,
#' honouring the `Retry-After` header when the server sends one.
#'
#' @param req An `httr2_request` object.
#' @param max_tries Maximum number of attempts, including the first one.
#' @param base_delay Base delay in seconds for exponential backoff.
#' @return The modified `httr2_request` object.
#' @export
with_retries <- function(req, max_tries = 4L, base_delay = 0.5) {
  httr2::req_retry(
    req,
    max_tries = max_tries,
    is_transient = \(resp) httr2::resp_status(resp) %in% c(429L, 500L, 502L, 503L, 504L),
    backoff = \(attempt) base_delay * 2^(attempt - 1) + stats::runif(1, 0, base_delay),
    after = retry_after_seconds
  )
}

#' Read the Retry-After header of a response in seconds
#'
#' Handles both the delta-seconds and the HTTP-date forms of the header.
#'
#' @param resp An `httr2_response` object.
#' @return A number of seconds, or `NA` if the header is missing or unparseable.
#' @keywords internal
retry_after_seconds <- function(resp) {
  value <- httr2::resp_header(resp, "Retry-After")
  if (is.null(value)) {
    return(NA_real_)
  }
  if (grepl("^[0-9]+$", value)) {
    return(as.numeric(value))
  }
  when <- httr2::parse_http_date(value)
  if (is.na(when)) {
    return(NA_real_)
  }
  max(0, as.numeric(difftime(when, Sys.time(), units = "secs")))
}

#' Perform a GET request against an endpoint
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param config A configuration list as returned by [clerk_config()].
#' @return The parsed JSON body as a list.
#' @family verbs
#' @export
clerk_get <- function(path, query = list(), config = clerk_config()) {
  query <- Filter(Negate(is.null), query)
  clerk_request(config) |>
    httr2::req_url_path_append(path) |>
    httr2::req_url_query(!!!query) |>
    with_retries(config$max_tries) |>
    httr2::req_error(body = api_error_body) |>
    httr2::req_perform() |>
    parse_json_body()
}

#' Perform a POST request against an endpoint
#'
#' @param path Endpoint path relative to the base URL.
#' @param body A list serialised to the JSON request body.
#' @param config A configuration list as returned by [clerk_config()].
#' @return The parsed JSON body as a list.
#' @family verbs
#' @export
clerk_post <- function(path, body = list(), config = clerk_config()) {
  clerk_request(config) |>
    httr2::req_url_path_append(path) |>
    httr2::req_body_json(body, auto_unbox = TRUE) |>
    with_retries(config$max_tries) |>
    httr2::req_error(body = api_error_body) |>
    httr2::req_perform() |>
    parse_json_body()
}

#' Perform a DELETE request against an endpoint
#'
#' @param path Endpoint path relative to the base URL.
#' @param config A configuration list as returned by [clerk_config()].
#' @return `TRUE`, invisibly, if the resource was deleted.
#' @family verbs
#' @export
clerk_delete <- function(path, config = clerk_config()) {
  resp <- clerk_request(config) |>
    httr2::req_url_path_append(path) |>
    httr2::req_method("DELETE") |>
    with_retries(config$max_tries) |>
    httr2::req_error(body = api_error_body) |>
    httr2::req_perform()
  invisible(httr2::resp_status(resp) %in% c(200L, 202L, 204L))
}

#' Parse a JSON response body
#'
#' Empty bodies (for example from 204 responses) become an empty list.
#'
#' @param resp An `httr2_response` object.
#' @return A list.
#' @keywords internal
parse_json_body <- function(resp) {
  if (!httr2::resp_has_body(resp)) {
    return(list())
  }
  type <- httr2::resp_content_type(resp)
  if (!identical(type, "application/json")) {
    stop("Expected a JSON response but got '", type, "'.", call. = FALSE)
  }
  httr2::resp_body_json(resp, simplifyVector = FALSE)
}

#' Extract an error message from an API error response
#'
#' @param resp An `httr2_response` object.
#' @return A character vector of message lines for [httr2::req_error()].
#' @keywords internal
api_error_body <- function(resp) {
  body <- tryCatch(
    httr2::resp_body_json(resp),
    error = \(e) NULL
  )
  if (is.null(body)) {
    return(NULL)
  }
  msg <- c(body$error$message, body$message)[1]
  code <- c(body$error$code, body$code)[1]
  c(
    if (!is.null(msg)) paste("API message:", msg),
    if (!is.null(code)) paste("Error code:", code)
  )
}

#' Perform GET requests for many endpoints in parallel
#'
#' Requests are built like [clerk_get()] and performed concurrently. Failed
#' requests do not abort the batch; their error messages are returned in the
#' `"failures"` attribute instead.
#'
#' @param paths Character vector of endpoint paths relative to the base URL.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param max_active Maximum number of simultaneous connections.
#' @param config A configuration list as returned by [clerk_config()].
#' @return A named list of parsed JSON bodies, `NULL` for failed requests.
#' @export
clerk_get_many <- function(paths, query = list(), max_active = 4L, config = clerk_config()) {
  if (length(paths) == 0L) {
    return(list())
  }
  query <- Filter(Negate(is.null), query)
  base <- clerk_request(config) |>
    httr2::req_url_query(!!!query) |>
    with_retries(config$max_tries) |>
    httr2::req_error(body = api_error_body)
  reqs <- lapply(paths, \(p) httr2::req_url_path_append(base, p))
  resps <- httr2::req_perform_parallel(reqs, on_error = "continue", max_active = max_active)
  out <- vector("list", length(paths))
  names(out) <- paths
  failures <- character()
  for (i in seq_along(resps)) {
    resp <- resps[[i]]
    if (inherits(resp, "error")) {
      failures[paths[[i]]] <- conditionMessage(resp)
      next
    }
    out[i] <- list(parse_json_body(resp))
  }
  if (length(failures) > 0L) {
    warning(length(failures), " of ", length(paths), " request(s) failed.", call. = FALSE)
  }
  attr(out, "failures") <- failures
  out
}

#' Fetch all pages of an offset-paginated endpoint
#'
#' Requests pages of `page_size` items until a short page is returned or
#' `max_pages` is reached.
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param page_size Number of items requested per page.
#' @param max_pages Upper bound on the number of requests made.
#' @param config A configuration list as returned by [clerk_config()].
#' @return A list of items, concatenated across pages.
#' @family paginators
#' @export
paginate_offset <- function(path, query = list(), page_size = 100L, max_pages = 50L,
                            config = clerk_config()) {
  pages <- vector("list", max_pages)
  for (i in seq_len(max_pages)) {
    q <- utils::modifyList(query, list(limit = page_size, offset = (i - 1L) * page_size))
    body <- clerk_get(path, q, config)
    pages[[i]] <- body$items
    if (length(body$items) < page_size) break
  }
  if (i == max_pages && length(pages[[i]]) == page_size) {
    warning("Stopped after ", max_pages, " pages; results may be incomplete.", call. = FALSE)
  }
  unlist(pages, recursive = FALSE)
}

#' Fetch all pages of a cursor-paginated endpoint
#'
#' Follows `next_cursor` in each response body until it is empty or
#' `max_pages` is reached.
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param page_size Number of items requested per page.
#' @param max_pages Upper bound on the number of requests made.
#' @param config A configuration list as returned by [clerk_config()].
#' @return A list of items, concatenated across pages.
#' @family paginators
#' @export
paginate_cursor <- function(path, query = list(), page_size = 100L, max_pages = 50L,
                            config = clerk_config()) {
  pages <- vector("list", max_pages)
  cursor <- NULL
  for (i in seq_len(max_pages)) {
    q <- utils::modifyList(query, list(limit = page_size, cursor = cursor))
    body <- clerk_get(path, q, config)
    pages[[i]] <- body$items
    cursor <- body$next_cursor
    if (is.null(cursor) || !nzchar(cursor)) break
  }
  if (!is.null(cursor) && nzchar(cursor)) {
    warning("Stopped after ", max_pages, " pages; results may be incomplete.", call. = FALSE)
  }
  unlist(pages, recursive = FALSE)
}

#' Fetch all pages by following RFC 8288 Link headers
#'
#' Uses the `rel="next"` link of each response to find the following page.
#'
#' @param path Endpoint path relative to the base URL.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param max_pages Upper bound on the number of requests made.
#' @param config A configuration list as returned by [clerk_config()].
#' @return A list of items, concatenated across pages.
#' @family paginators
#' @export
paginate_links <- function(path, query = list(), max_pages = 50L, config = clerk_config()) {
  pages <- vector("list", max_pages)
  req <- clerk_request(config) |>
    httr2::req_url_path_append(path) |>
    httr2::req_url_query(!!!Filter(Negate(is.null), query)) |>
    with_retries(config$max_tries)
  next_url <- NULL
  for (i in seq_len(max_pages)) {
    if (!is.null(next_url)) {
      req <- httr2::req_url(req, next_url)
    }
    resp <- httr2::req_perform(req)
    pages[[i]] <- parse_json_body(resp)
    next_url <- httr2::resp_link_url(resp, "next")
    if (is.null(next_url)) break
  }
  if (!is.null(next_url)) {
    warning("Stopped after ", max_pages, " pages; results may be incomplete.", call. = FALSE)
  }
  unlist(pages, recursive = FALSE)
}

#' Convert a list of API items to a data.table
#'
#' Scalar fields become columns; nested fields are kept as list columns unless
#' listed in `flatten`, in which case their sub-fields are prefixed and spread.
#'
#' @param items A list of items as returned by a paginator.
#' @param flatten Character vector of nested fields to spread into columns.
#' @return A data.table with one row per item.
#' @export
items_to_table <- function(items, flatten = character()) {
  if (length(items) == 0L) {
    return(data.table::data.table())
  }
  rows <- lapply(items, \(item) {
    for (field in intersect(flatten, names(item))) {
      sub <- item[[field]]
      item[[field]] <- NULL
      if (length(sub) > 0L) {
        names(sub) <- paste(field, names(sub), sep = "_")
        item <- c(item, sub)
      }
    }
    lapply(item, \(v) if (is.null(v)) NA else if (length(v) > 1L) list(v) else v)
  })
  data.table::rbindlist(rows, use.names = TRUE, fill = TRUE)
}

#' Fetch an endpoint with the paginator matching its style
#'
#' @param path Endpoint path relative to the base URL.
#' @param style Pagination style used by the endpoint.
#' @param ... Passed to the selected paginator.
#' @return A data.table with one row per item.
#' @export
fetch_all <- function(path, style = c("cursor", "offset", "links"), ...) {
  style <- match.arg(style)
  items <- switch(style,
    cursor = paginate_cursor(path, ...),
    offset = paginate_offset(path, ...),
    links = paginate_links(path, ...)
  )
  items_to_table(items)
}

#' Fetch only items updated since the previous run
#'
#' Reads the high-water mark from `state_file`, requests items with
#' `updated_since` set to it, and saves the newest `updated_at` value seen so
#' the next run continues from there.
#'
#' @param path Endpoint path relative to the base URL.
#' @param state_file Path to a small JSON file holding the high-water mark.
#' @param query A named list of query parameters; `NULL` entries are dropped.
#' @param overlap Seconds subtracted from the mark to tolerate clock skew.
#' @param config A configuration list as returned by [clerk_config()].
#' @return A data.table of new or updated items.
#' @export
fetch_incremental <- function(path, state_file, query = list(), overlap = 60,
                              config = clerk_config()) {
  state <- if (file.exists(state_file)) {
    jsonlite::read_json(state_file, simplifyVector = TRUE)
  } else {
    list()
  }
  since <- state[[path]]
  if (!is.null(since)) {
    since_time <- as.POSIXct(since, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC") - overlap
    query$updated_since <- format(since_time, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  }
  items <- paginate_cursor(path, query = query, config = config)
  out <- items_to_table(items)
  if (nrow(out) == 0L) {
    message("No updates for ", path, if (!is.null(since)) paste(" since", since))
    return(out)
  }
  if (!"updated_at" %in% names(out)) {
    stop("Items from ", path, " have no `updated_at` field.", call. = FALSE)
  }
  # ISO 8601 strings in UTC sort correctly as text
  state[[path]] <- max(out$updated_at, na.rm = TRUE)
  dir.create(dirname(state_file), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(state, state_file, auto_unbox = TRUE, pretty = TRUE)
  data.table::setorder(out, updated_at)
  out
}

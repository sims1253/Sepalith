#' Read pagination state from a response
#'
#' @param body Parsed response body (a nested list).
#' @param headers Named character vector of response headers, lower-case names.
#' @param strategy One of `"page"`, `"offset"`, `"cursor"`, `"link"` or
#'   `"token"`.
#' @return A list whose elements depend on `strategy`: `page`, `perPage`,
#'   `totalPages` for `"page"`; `offset`, `limit`, `total` for `"offset"`;
#'   `cursor`, `hasMore` for `"cursor"`; `nextUrl` for `"link"`; `token`
#'   for `"token"`. Absent values are `NA`.
#' @export
pageInfo = function(body, headers = character(0),
                    strategy = c("page", "offset", "cursor", "link", "token")) {
  strategy = match.arg(strategy)
  orNA = function(x, na) if (is.null(x)) na else x

  info = switch(strategy,
    page = {
      meta = orNA(body$meta, list())
      list(
        page = as.integer(orNA(meta$page, NA_integer_)),
        perPage = as.integer(orNA(meta$per_page, NA_integer_)),
        totalPages = as.integer(orNA(meta$total_pages, NA_integer_))
      )
    },
    offset = {
      list(
        offset = as.integer(orNA(body$offset, NA_integer_)),
        limit = as.integer(orNA(body$limit, NA_integer_)),
        total = as.integer(orNA(body$total, NA_integer_))
      )
    },
    cursor = {
      meta = orNA(body$meta, list())
      list(
        cursor = as.character(orNA(meta$next_cursor, NA_character_)),
        hasMore = isTRUE(meta$has_more)
      )
    },
    link = {
      link = if ("link" %in% names(headers)) headers[["link"]] else ""
      hit = regmatches(link, regexec('<([^>]*)>; *rel="?next"?', link))[[1]]
      list(nextUrl = if (length(hit) == 2L) hit[[2]] else NA_character_)
    },
    token = {
      list(token = as.character(orNA(body$nextPageToken, NA_character_)))
    }
  )
  return(info)
}

#' Is there another page?
#'
#' @param info Pagination state from [pageInfo()].
#' @param strategy One of `"page"`, `"offset"`, `"cursor"`, `"link"` or
#'   `"token"`.
#' @param nItems Number of items on the current page.
#' @return `TRUE` or `FALSE`.
#' @export
hasMorePages = function(info, strategy = c("page", "offset", "cursor", "link", "token"),
                        nItems = NA_integer_) {
  strategy = match.arg(strategy)
  more = switch(strategy,
    page = {
      if (is.na(info$totalPages)) {
        !is.na(nItems) && !is.na(info$perPage) && nItems >= info$perPage
      } else {
        info$page < info$totalPages
      }
    },
    offset = {
      if (is.na(info$total)) {
        !is.na(nItems) && !is.na(info$limit) && nItems >= info$limit
      } else {
        info$offset + info$limit < info$total
      }
    },
    cursor = {
      info$hasMore && !is.na(info$cursor) && nzchar(info$cursor)
    },
    link = {
      !is.na(info$nextUrl) && nzchar(info$nextUrl)
    },
    token = {
      !is.na(info$token) && nzchar(info$token)
    }
  )
  return(isTRUE(more))
}

#' Build the query for the next page
#'
#' @param query The named list of query parameters used for the current page.
#' @param info Pagination state from [pageInfo()].
#' @param strategy One of `"page"`, `"offset"`, `"cursor"`, `"link"` or
#'   `"token"`.
#' @return The updated query list.
#' @export
nextPageQuery = function(query, info, strategy = c("page", "offset", "cursor", "link", "token")) {
  strategy = match.arg(strategy)
  query = switch(strategy,
    page = {
      current = if (is.na(info$page)) 1L else info$page
      query$page = current + 1L
      query
    },
    offset = {
      current = if (is.na(info$offset)) 0L else info$offset
      query$offset = current + info$limit
      query
    },
    cursor = {
      query$cursor = info$cursor
      query
    },
    link = {
      # Copy every parameter of the next URL into the query
      qs = sub("^[^?]*\\?", "", info$nextUrl)
      pairs = strsplit(strsplit(qs, "&", fixed = TRUE)[[1]], "=", fixed = TRUE)
      for (p in pairs) {
        query[[utils::URLdecode(p[1])]] = utils::URLdecode(p[2])
      }
      query
    },
    token = {
      query$pageToken = info$token
      query
    }
  )
  return(query)
}

#' Fetch every page of a listing
#'
#' @param fetch A function taking a query list and returning a list with
#'   elements `body` and `headers`.
#' @param query The named list of query parameters for the first page.
#' @param strategy One of `"page"`, `"offset"`, `"cursor"`, `"link"` or
#'   `"token"`.
#' @param itemsField Name of the body element holding the items.
#' @param maxPages Stop after this many pages.
#' @return A list with elements `items` (all items, in order), `pages` (the
#'   number of pages fetched) and `complete` (`FALSE` when `maxPages` was hit).
#' @export
paginateAll = function(fetch, query = list(),
                       strategy = c("page", "offset", "cursor", "link", "token"),
                       itemsField = "items", maxPages = 100L) {
  strategy = match.arg(strategy)
  items = list()
  pages = 0L
  complete = FALSE
  while (pages < maxPages) {
    resp = fetch(query)
    pages = pages + 1L
    pageItems = resp$body[[itemsField]]
    items = c(items, pageItems)
    info = pageInfo(resp$body, resp$headers, strategy)
    if (!hasMorePages(info, strategy, nItems = length(pageItems))) {
      complete = TRUE
      break
    }
    query = nextPageQuery(query, info, strategy)
  }

  return(list(items = items, pages = pages, complete = complete))
}

#' Estimate the number of pages left
#'
#' @param info Pagination state from [pageInfo()].
#' @param strategy One of `"page"`, `"offset"`, `"cursor"`, `"link"` or
#'   `"token"`.
#' @return A single integer, or `NA` when the strategy gives no totals.
#' @export
pagesRemaining = function(info, strategy = c("page", "offset", "cursor", "link", "token")) {
  strategy = match.arg(strategy)
  left = switch(strategy,
    page = {
      if (is.na(info$totalPages) || is.na(info$page)) {
        NA_integer_
      } else {
        max(info$totalPages - info$page, 0L)
      }
    },
    offset = {
      if (is.na(info$total) || is.na(info$limit)) {
        NA_integer_
      } else {
        max(ceiling((info$total - info$offset - info$limit) / info$limit), 0L)
      }
    },
    cursor = {
      if (isTRUE(info$hasMore)) NA_integer_ else 0L
    },
    link = {
      if (is.na(info$nextUrl)) 0L else NA_integer_
    },
    token = {
      if (is.na(info$token)) 0L else NA_integer_
    }
  )
  return(as.integer(left))
}

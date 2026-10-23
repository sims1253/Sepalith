p = pageInfo(list(meta = list(page = 2, per_page = 10, total_pages = 5)), strategy = "page")
stopifnot(identical(p, list(page = 2L, perPage = 10L, totalPages = 5L)))
o = pageInfo(list(offset = 20, limit = 10, total = 35), strategy = "offset")
stopifnot(identical(o, list(offset = 20L, limit = 10L, total = 35L)))
cu = pageInfo(list(meta = list(next_cursor = "abc", has_more = TRUE)), strategy = "cursor")
stopifnot(identical(cu, list(cursor = "abc", hasMore = TRUE)))
lk = pageInfo(list(), c(link = '<http://h/x?page=3&q=a>; rel="next"'), strategy = "link")
stopifnot(identical(lk$nextUrl, "http://h/x?page=3&q=a"))
stopifnot(is.na(pageInfo(list(), strategy = "link")$nextUrl))
tk = pageInfo(list(nextPageToken = "t2"), strategy = "token")
stopifnot(identical(tk$token, "t2"), is.na(pageInfo(list(), strategy = "token")$token))
stopifnot(is.na(pageInfo(list(), strategy = "page")$totalPages))

stopifnot(hasMorePages(p, "page"), !hasMorePages(list(page = 5L, perPage = 10L, totalPages = 5L)))
stopifnot(hasMorePages(list(page = 1L, perPage = 10L, totalPages = NA), "page", nItems = 10L))
stopifnot(hasMorePages(o, "offset"), !hasMorePages(list(offset = 30L, limit = 10L, total = 35L),
  "offset"))
stopifnot(hasMorePages(list(offset = 0L, limit = 5L, total = NA), "offset", nItems = 5L))
stopifnot(hasMorePages(cu, "cursor"), !hasMorePages(list(cursor = NA, hasMore = TRUE), "cursor"))
stopifnot(hasMorePages(lk, "link"), !hasMorePages(list(nextUrl = NA), "link"))
stopifnot(hasMorePages(tk, "token"), !hasMorePages(list(token = ""), "token"))

stopifnot(identical(nextPageQuery(list(q = "a"), p, "page"), list(q = "a", page = 3L)))
stopifnot(identical(nextPageQuery(list(), o, "offset")$offset, 30L))
stopifnot(identical(nextPageQuery(list(), cu, "cursor")$cursor, "abc"))
stopifnot(identical(nextPageQuery(list(q = "z"), lk, "link"), list(q = "a", page = "3")))
stopifnot(identical(nextPageQuery(list(), tk, "token")$pageToken, "t2"))

pagesData = list(list(1, 2), list(3, 4), list(5))
fetchPage = function(query) {
  page = if (is.null(query$page)) 1L else query$page
  list(body = list(items = pagesData[[page]],
    meta = list(page = page, per_page = 2L, total_pages = 3L)), headers = character(0))
}
all = paginateAll(fetchPage, strategy = "page")
stopifnot(identical(unlist(all$items), c(1, 2, 3, 4, 5)), all$pages == 3L, all$complete)
part = paginateAll(fetchPage, strategy = "page", maxPages = 2L)
stopifnot(length(part$items) == 4L, !part$complete)
fetchTok = function(query) {
  if (is.null(query$pageToken)) {
    list(body = list(data = list("a"), nextPageToken = "n"), headers = character(0))
  } else {
    list(body = list(data = list("b")), headers = character(0))
  }
}
tok = paginateAll(fetchTok, strategy = "token", itemsField = "data")
stopifnot(identical(unlist(tok$items), c("a", "b")), tok$pages == 2L)

stopifnot(identical(pagesRemaining(p, "page"), 3L))
noTotals = list(page = NA, perPage = 1L, totalPages = NA)
stopifnot(identical(pagesRemaining(noTotals, "page"), NA_integer_))
stopifnot(identical(pagesRemaining(list(offset = 0L, limit = 10L, total = 35L), "offset"), 3L))
stopifnot(identical(pagesRemaining(o, "offset"), 1L))
stopifnot(identical(pagesRemaining(cu, "cursor"), NA_integer_))
stopifnot(identical(pagesRemaining(list(cursor = NA, hasMore = FALSE), "cursor"), 0L))
stopifnot(identical(pagesRemaining(list(nextUrl = NA), "link"), 0L))
stopifnot(identical(pagesRemaining(tk, "token"), NA_integer_))

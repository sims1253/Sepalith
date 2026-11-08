q <- build_search_query(q = " cats ", page = 2, per_page = 50, sort = "stars", order = "desc",
  since = as.Date("2024-03-01"), fields = c("id", "name", "id"))
stopifnot(identical(names(q), c("q", "page", "per_page", "sort", "order", "since", "fields")))
stopifnot(identical(q$q, "cats"), identical(q$page, "2"), identical(q$per_page, "50"))
stopifnot(identical(q$since, "2024-03-01T00:00:00Z"), identical(q$fields, "id,name"))
stopifnot(identical(build_search_query(), list()))
stopifnot(inherits(try(build_search_query(per_page = 500), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_search_query(order = "up"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_search_query(page = 0), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_search_query(sort = ""), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_search_query(since = "2024"), silent = TRUE), "try-error"))
stopifnot(inherits(try(build_search_query(fields = character(0)), silent = TRUE), "try-error"))

stopifnot(identical(encode_query_pair("id", c(1, 2)), "id=1&id=2"))
stopifnot(identical(encode_query_pair("id", c(1, 2), "comma"), "id=1,2"))
stopifnot(identical(encode_query_pair("id", c(1, 2), "space"), "id=1%202"))
stopifnot(identical(encode_query_pair("id", c(1, 2), "pipe"), "id=1|2"))
stopifnot(identical(encode_query_pair("id", c(1, 2), "brackets"), "id%5B%5D=1&id%5B%5D=2"))
stopifnot(identical(encode_query_pair("ok", TRUE), "ok=true"))
stopifnot(identical(encode_query_pair("d", as.Date("2024-01-02")), "d=2024-01-02"))
stopifnot(identical(encode_query_pair("x", NA), ""))

r <- check_query_params(list(limit = "10", offset = 0, q = "abc", lang = "en",
  from = "2024-01-01", to = "2024-02-01"))
stopifnot(length(r$problems) == 0L)
r <- check_query_params(list(limit = 0, offset = -1, q = "", lang = "xx", from = "nope",
  to = "nope", foo = 1))
stopifnot(length(r$problems) == 7L)
stopifnot(identical(r$problems[1], "`foo` is not a known parameter."))
stopifnot(identical(r$problems[5], "`lang` is not an allowed value."))
stopifnot(identical(r$problems[7], "`to` is not a valid date."))

u <- build_request_url("https://api.example.org/", c("v1", "repos", "a b"),
  list(q = "x y", page = 2, skip = NULL), fragment = "top")
stopifnot(identical(u, "https://api.example.org/v1/repos/a%20b?q=x%20y&page=2#top"))
stopifnot(identical(build_request_url("http://h"), "http://h"))
stopifnot(identical(build_request_url("http://h", query = list(id = 1:2), style = "comma"),
  "http://h?id=1,2"))
stopifnot(inherits(try(build_request_url("ftp://h"), silent = TRUE), "try-error"))

p <- parse_query_string("?a=1&b=2.5&c=TRUE&d=2024-01-05&e=x,y&a=3&f=hi%20there",
  types = c(a = "integer", b = "numeric", c = "logical", d = "date", e = "csv"))
stopifnot(identical(p$a, c(1L, 3L)), identical(p$b, 2.5), identical(p$c, TRUE))
stopifnot(identical(p$d, as.Date("2024-01-05")), identical(p$e, c("x", "y")))
stopifnot(identical(unname(p$f), "hi there"))
stopifnot(identical(parse_query_string(""), list()))
stopifnot(inherits(try(parse_query_string("a=1", c(a = "bad")), silent = TRUE), "try-error"))

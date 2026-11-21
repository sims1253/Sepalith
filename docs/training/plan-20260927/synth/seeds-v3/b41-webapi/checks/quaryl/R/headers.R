h <- build_request_headers(token = "abc", content_type = "application/json",
  request_id = "r-1", if_none_match = "e1")
stopifnot(identical(names(h), c("Authorization", "Accept", "User-Agent", "Content-Type",
  "X-Request-Id", "If-None-Match")))
stopifnot(identical(h[["Authorization"]], "Bearer abc"), identical(h[["User-Agent"]], "quaryl"))
stopifnot(identical(build_request_headers(accept = NULL, user_agent = NULL), character(0)))
for (arg in c("token", "accept", "user_agent", "content_type", "request_id", "if_none_match")) {
  args <- stats::setNames(list(""), arg)
  stopifnot(inherits(try(do.call(build_request_headers, args), silent = TRUE), "try-error"))
}

stopifnot(identical(base64_encode(charToRaw("Man")), "TWFu"))
stopifnot(identical(base64_encode(charToRaw("Ma")), "TWE="))
stopifnot(identical(base64_encode(charToRaw("M")), "TQ=="))
stopifnot(identical(base64_encode(raw(0)), ""))

stopifnot(identical(auth_header(list(token = "t"))[["Authorization"]], "Bearer t"))
b <- auth_header(list(user = "user", password = "pass"), "basic")
stopifnot(identical(b[["Authorization"]], "Basic dXNlcjpwYXNz"))
stopifnot(identical(auth_header(list(token = "t"), "token")[["Authorization"]], "token t"))
stopifnot(identical(auth_header(list(key = "k"), "apikey"), c("X-Api-Key" = "k")))
stopifnot(identical(auth_header(list(key = "k", header = "Api"), "apikey"), c(Api = "k")))
stopifnot(identical(auth_header(scheme = "none"), character(0)))
stopifnot(inherits(try(auth_header(list(user = "u"), "basic"), silent = TRUE), "try-error"))

r <- parse_response_headers(c("HTTP/1.1 200 OK", "Content-Type: application/json; charset=utf-8",
  "content-length: 42", "ETag: W/\"abc\"", "Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT",
  "Retry-After: 30", "Link: <http://x?page=2>; rel=\"next\""))
stopifnot(identical(r$content_type, "application/json"), identical(r$content_length, 42L))
stopifnot(identical(r$etag, "abc"), identical(r$retry_after, 30))
stopifnot(identical(format(r$last_modified, "%Y-%m-%d %H:%M"), "2015-10-21 07:28"))
stopifnot(identical(r$link, "<http://x?page=2>; rel=\"next\""))
e <- parse_response_headers(character(0))
stopifnot(is.na(e$content_type), is.na(e$content_length), is.na(e$etag), is.na(e$last_modified))
stopifnot(is.na(e$retry_after), is.na(e$link))

l <- parse_link_header(paste0("<http://a?p=3>; rel=\"next\", <http://a?p=1>; rel=\"prev\", ",
  "<http://a?p=1>; rel=\"first\", <http://a?p=9>; rel=\"last\""))
stopifnot(identical(l[["next"]], "http://a?p=3"), identical(l$prev, "http://a?p=1"))
stopifnot(identical(l$first, "http://a?p=1"), identical(l$last, "http://a?p=9"))
stopifnot(all(is.na(unlist(parse_link_header(NA)))))
stopifnot(is.na(parse_link_header("<http://a>; rel=\"next\"")$last))

stopifnot(identical(check_content_type("application/vnd.api+json"), "application/vnd.api+json"))
stopifnot(identical(check_content_type("text/xml", "xml"), "text/xml"))
stopifnot(identical(check_content_type("text/csv", "csv"), "text/csv"))
stopifnot(identical(check_content_type("text/html", "text"), "text/html"))
stopifnot(identical(check_content_type("image/png", "binary"), "image/png"))
stopifnot(inherits(try(check_content_type("text/html", "json"), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_content_type(NA), silent = TRUE), "try-error"))

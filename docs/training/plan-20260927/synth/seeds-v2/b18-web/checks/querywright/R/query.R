# skipped: url_modify_query(), url_drop_query(), url_keep_query() and
# url_sort_query() need url_parts() / url_assemble() from url.R
stopifnot(
  identical(encode_component(c("a b", "x&y")), c("a%20b", "x%26y")),
  identical(encode_component("a b", space_as_plus = TRUE), "a+b"),
  identical(decode_component(c("red+shoes", "50%25")), c("red shoes", "50%"))
)

stopifnot(
  identical(
    query_build(list(q = "red shoes", page = 2, debug = NULL, tag = c("a&b", NA, "c"))),
    "q=red%20shoes&page=2&tag=a%26b&tag=c"
  ),
  identical(query_build(list(on = TRUE, d = as.Date("2026-03-01"))), "on=true&d=2026-03-01"),
  identical(query_build(list(gone = NA)), ""),
  inherits(try(query_build(list(1)), silent = TRUE), "try-error")
)

params <- query_parse("?tag=a%26b&tag=c&q=red+shoes&flag&=orphan&eq=a=b")
stopifnot(
  identical(params$tag, c("a&b", "c")),
  identical(params$flag, ""),
  identical(params$eq, "a=b"),
  identical(names(params), c("tag", "q", "flag", "eq")),
  identical(query_parse(""), list())
)

round <- list(q = "50% off", tag = c("x", "y"))
stopifnot(identical(query_parse(query_build(round)), round))

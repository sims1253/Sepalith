# skipped: md.italic, md.bold, md.link and md.to.html need escape.* from
# R/escape.R
# skipped: needs stringi and magrittr: slugify

# md.code
stopifnot(identical(md.code("x <- 1"), "`x <- 1`"))
stopifnot(identical(md.code("a `b` c"), "``a `b` c``"))
stopifnot(identical(md.code("`x`"), "`` `x` ``"))

# md.list
stopifnot(identical(md.list(c("a", "b")), "- a\n- b"))
stopifnot(identical(md.list(c("a", "b"), ordered = TRUE, indent = 2L), "  1. a\n  2. b"))
stopifnot(identical(md.list(character()), ""))

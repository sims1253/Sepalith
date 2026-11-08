# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

src <- c(
  "f <- function(x, y = 2L) {",
  "  z = x[[1]] + 0x1F; T",
  "  g(a = \"s\", NULL) # hi",
  "  h <- y -> w; base::sum(1e3, 1, \"s\")",
  "  #' doc",
  "}"
)

ts <- token_summary(src)
stopifnot(identical(names(ts), c("symbols", "calls", "numbers", "strings", "comments")))
stopifnot(identical(ts$symbols$text, c("f", "z", "x", "T", "h", "y", "w")))
stopifnot(ts$symbols$n[ts$symbols$text == "y"] == 1L, ts$symbols$line[3] == 2L)
stopifnot(identical(ts$calls$text, c("g", "sum")), identical(ts$calls$line, c(3L, 4L)))
stopifnot(identical(ts$numbers$text, c("2L", "1", "0x1F", "1e3")))
stopifnot(identical(ts$strings$n, 2L), identical(ts$comments$text, c("# hi", "#' doc")))
stopifnot(nrow(token_summary("1 + 2")$symbols) == 0L)

dt <- describe_tokens(src)
stopifnot(identical(dt$detail[dt$text == "<-"], c("left", "left")))
stopifnot(identical(dt$detail[dt$text == "->"], "right"))
stopifnot(identical(dt$detail[dt$token == "EQ_ASSIGN"], "equals"))
stopifnot(identical(dt$detail[dt$token == "NUM_CONST"], c("integer", rep("double", 4))))
stopifnot(all(dt$detail[dt$token == "STR_CONST"] == "character"))
stopifnot(identical(dt$detail[dt$token == "NULL_CONST"], "NULL"))
stopifnot(identical(dt$detail[dt$kind == "comment"], c("plain", "roxygen")))
stopifnot(identical(dt$detail[dt$kind == "call"], c("bare", "namespaced")))
stopifnot(identical(dt$detail[dt$text == "T"], "abbreviation"), dt$kind[3] == "other")
stopifnot(identical(describe_tokens("a <<- 1; 2 ->> b")$detail[c(2, 6)],
  c("superassign", "superassign")))

bd <- bracket_depth(src)
stopifnot(identical(bd$kind, c("paren", "brace", "square")))
stopifnot(identical(bd$opened, c(3L, 1L, 1L)), identical(bd$max_depth, c(1L, 1L, 2L)))
stopifnot(identical(bd$max_line, c(1L, 1L, 2L)))
bd2 <- bracket_depth(c("{", "  m[i, f(g(1))]", "}"))
stopifnot(identical(bd2$max_depth, c(2L, 1L, 1L)), identical(bd2$max_line, c(2L, 1L, 2L)))

sr <- symbol_roles(src)
stopifnot(identical(sr$assigned, c("f", "h", "w", "z")), identical(sr$formals, c("x", "y")))
stopifnot(identical(sr$arguments, "a"), identical(sr$called, c("g", "sum")))
stopifnot(identical(sr$packages, "base"), identical(sr$used, c("T", "x", "y")))
stopifnot(identical(symbol_roles("`my var` <- 1")$assigned, "my var"))

stopifnot(identical(extract_tokens(src, "strings"), c(`3` = "s", `4` = "s")))
stopifnot(identical(unname(extract_tokens(src, "numbers")), c(2, 1, 31, 1000, 1)))
stopifnot(identical(extract_tokens(src, "comments"), c(`3` = "hi", `5` = "doc")))
stopifnot(identical(extract_tokens(src, "calls"), c(`3` = "g", `4` = "sum")))
stopifnot(identical(extract_tokens(src, "symbols", unique = TRUE)[1:3],
  c(`1` = "f", `2` = "z", `2` = "x")))
stopifnot(length(extract_tokens(src, "strings", unique = TRUE)) == 1L)
stopifnot(inherits(tryCatch(extract_tokens(src, "ops"), error = identity), "error"))

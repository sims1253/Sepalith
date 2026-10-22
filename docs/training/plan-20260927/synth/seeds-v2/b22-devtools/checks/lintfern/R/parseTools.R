src <- tempfile(fileext = ".R")
writeLines(c(
  "f <- function(x, n = 2) x^n",
  "g = function(a) {",
  "  # helper",
  "  lapply(a, function(z) z)",
  "}"
), src)

# readSourceTokens
pd <- readSourceTokens(src)
stopifnot(is.data.frame(pd), all(pd$file == src), !is.unsorted(pd$line1))
bad <- tempfile(fileext = ".R")
writeLines("f <- function( {", bad)
stopifnot(is.null(suppressWarnings(readSourceTokens(bad))))

# tokenChildren / tokenDescendants / tokenParent
top <- pd$id[pd$parent == 0 & pd$token == "expr"][1]
kids <- tokenChildren(pd, top)
stopifnot(identical(kids$token, c("expr", "LEFT_ASSIGN", "expr")))
stopifnot(all(tokenDescendants(pd, top)$line1 == 1L))
stopifnot(is.null(tokenParent(pd, top)), tokenParent(pd, kids$id[1])$id == top)

# findFunctionDefs
defs <- findFunctionDefs(pd)
stopifnot(nrow(defs) == 3L, identical(defs$name, c("f", "g", NA)))
stopifnot(identical(defs$line2, c(1L, 5L, 4L)))

# functionFormals / functionBodySymbols / functionCodeLines
stopifnot(identical(functionFormals(pd, defs$id[1])$text, c("x", "n")))
stopifnot(identical(sort(functionBodySymbols(pd, defs$id[1])), c("n", "x")))
stopifnot("lapply" %in% functionBodySymbols(pd, defs$id[2]))
stopifnot(functionCodeLines(pd, defs$id[2]) == 3L, functionCodeLines(pd, defs$id[1]) == 1L)

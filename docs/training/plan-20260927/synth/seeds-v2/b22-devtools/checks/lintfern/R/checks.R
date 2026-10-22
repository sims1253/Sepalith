# skipped: checkUnusedArguments and checkLongFunctions need findFunctionDefs()
# and friends from parseTools.R, which is not sourced here

# lintHits
h <- lintHits(c(3, 5), c(1, 2), "msg")
stopifnot(identical(h$line, c(3L, 5L)), identical(h$message, c("msg", "msg")))
stopifnot(nrow(lintHits()) == 0L)
stopifnot(inherits(try(lintHits(1:2, 1L), silent = TRUE), "try-error"))

# checkTodoComments
op <- options(keep.parse.data = TRUE)
code <- c("# TODO: fix", "x <- 1 # fixme", "# fine")
pd <- utils::getParseData(parse(text = code, keep.source = TRUE))
options(op)
hits <- checkTodoComments(pd)
stopifnot(identical(hits$message, c("TODO: fix", "Bare FIXME comment.")))
stopifnot(identical(hits$line, 1:2))
stopifnot(nrow(checkTodoComments(pd, tags = "HACK")) == 0L)

# checkLineLength
ll <- checkLineLength(c("ab", "abcdef"), maxWidth = 4L)
stopifnot(identical(ll$line, 2L), identical(ll$column, 5L))

# checkTrailingWhitespace
tw <- checkTrailingWhitespace(c("a <- 1  ", "b", "c\t"))
stopifnot(identical(tw$line, c(1L, 3L)), identical(tw$column, c(7L, 2L)))

# checkTabIndent
ti <- checkTabIndent(c("\tx", "  \ty", "z\t"))
stopifnot(identical(ti$line, 1:2), identical(ti$column, c(1L, 3L)))

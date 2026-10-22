# skipped: lintFile and lintDir need readSourceTokens() and defaultRules() from
# parseTools.R and rules.R, which are not sourced here

rule <- structure(
  list(
    name = "short", input = "lines", severity = "style",
    check = function(lines) {
      bad <- which(nchar(lines) < 2)
      data.frame(line = bad, column = rep(1L, length(bad)), message = "Too short.")
    }
  ),
  class = "lintRule"
)
ctx <- list(file = "a.R", lines = c("x", "yy", "z"), pd = NULL)

# applyRule
hits <- applyRule(rule, ctx)
stopifnot(identical(hits$line, c(1L, 3L)), all(hits$rule == "short"), all(hits$file == "a.R"))
stopifnot(is.null(applyRule(modifyList(rule, list(input = "parse")), ctx)))
nothing <- rule
nothing$check <- function(lines) NULL
stopifnot(is.null(applyRule(nothing, ctx)))

# newLintResult
res <- newLintResult(list(hits[2:1, ]), files = "a.R")
stopifnot(inherits(res, "lintResult"), identical(res$line, c(1L, 3L)))
empty <- newLintResult(list(), files = c("a.R", "b.R"))
stopifnot(nrow(empty) == 0L, identical(attr(empty, "files"), c("a.R", "b.R")))

# format / print / summary
stopifnot(identical(format(res)[1], "a.R:1:1: style: Too short. [short]"))
stopifnot(identical(utils::capture.output(print(empty)), "No lints found in 2 file(s)."))
out <- utils::capture.output(print(res))
stopifnot(identical(out[length(out)], "2 lint(s) in 1 file(s)."))
s <- summary(res)
stopifnot(identical(s$rule, "short"), s$hits == 2L, s$files == 1L)

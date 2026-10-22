# newLintRule / validateLintRule
r <- newLintRule("my_rule", function(lines) NULL, input = "lines")
stopifnot(inherits(r, "lintRule"), r$severity == "warning", r$input == "lines")
stopifnot(inherits(try(newLintRule("BadName", identity), silent = TRUE), "try-error"))
stopifnot(inherits(try(newLintRule("no_args", function() 1), silent = TRUE), "try-error"))
stopifnot(inherits(newLintRule("prim", length), "lintRule"))

# format.lintRule / print.lintRule
stopifnot(identical(format(r), "<lintRule my_rule> [lines, warning]"))
out <- utils::capture.output(res <- print(r))
stopifnot(identical(res, r), identical(out, "<lintRule my_rule> [lines, warning]"))

# rule constructors and defaultRules
stopifnot(ruleLongFunctions()$severity == "style", ruleUnusedArguments()$input == "parse")
stopifnot(identical(ruleLineLength(100L)$name, "line_length"))
rules <- defaultRules()
stopifnot(identical(
  names(rules),
  c("unused_arguments", "long_functions", "todo_comments", "line_length")
))

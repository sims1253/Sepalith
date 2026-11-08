err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)

good <- list(rules = c("assign_eq", "semicolon"), exclude = "R/zzz.R", max_line_length = 100L,
  indent = 2L, naming = "snake_case", severity = "style")
stopifnot(identical(validate_lint_config(good), good))
stopifnot(identical(validate_lint_config(list()), list()))
stopifnot(identical(err(validate_lint_config(list(rules = 1))),
  "`rules` must be a character vector of rule names."))
stopifnot(identical(err(validate_lint_config(list(exclude = TRUE))),
  "`exclude` must be a character vector of file patterns."))
msg <- err(validate_lint_config(list(max_line_length = 20)))
stopifnot(grepl("`max_line_length` must be", msg))
stopifnot(grepl("`indent` must be", err(validate_lint_config(list(indent = 2.5)))))
stopifnot(grepl("`naming` must be", err(validate_lint_config(list(naming = "kebab")))))
stopifnot(grepl("`severity` must be", err(validate_lint_config(list(severity = "fatal")))))
two <- err(validate_lint_config(list(indent = 0, naming = c("snake_case", "camelCase"))))
stopifnot(length(strsplit(two, "\n")[[1]]) == 2L)
stopifnot(identical(err(validate_lint_config(list(colour = 1))), "Unknown config fields: colour."))

path <- tempfile(fileext = ".dcf")
write_lint_config(good, path)
stopifnot(identical(readLines(path)[1], "rules: assign_eq, semicolon"))
stopifnot(identical(read_lint_config(path), good))
write_lint_config(list(indent = 4L), path)
stopifnot(identical(readLines(path), "indent: 4"))
stopifnot(identical(read_lint_config(path), list(indent = 4L)))
writeLines(c("naming: camelCase", "exclude: a.R,  b.R", "max_line_length: 120"), path)
stopifnot(identical(read_lint_config(path),
  list(exclude = c("a.R", "b.R"), max_line_length = 120L, naming = "camelCase")))
writeLines("severity: loud", path)
stopifnot(grepl("`severity` must be", err(read_lint_config(path))))
stopifnot(grepl("does not exist", err(read_lint_config(tempfile()))))
unlink(path)

over <- list(rules = c("-semicolon", "true_false"), exclude = "R/aaa.R", max_line_length = 80L,
  indent = 4L, naming = "camelCase", severity = "info")
m <- merge_lint_config(good, over)
stopifnot(identical(m$rules, c("assign_eq", "true_false")))
stopifnot(identical(m$exclude, c("R/zzz.R", "R/aaa.R")))
stopifnot(m$max_line_length == 80L, m$indent == 4L, m$naming == "camelCase", m$severity == "style")
stopifnot(identical(merge_lint_config(good, list(severity = "error"))$severity, "error"))
stopifnot(identical(merge_lint_config(good, list()), good))

fc <- format_lint_config(good)
stopifnot(length(fc) == 6L, identical(fc[1], "rules           : assign_eq, semicolon"))
stopifnot(identical(fc[3], "max_line_length : 100"), identical(fc[6], "severity        : style"))
fd <- format_lint_config(list(indent = 2L))
stopifnot(identical(fd[2], "exclude         : <default>"), identical(fd[4], "indent          : 2"))
stopifnot(identical(fd[5], "naming          : <default>"))

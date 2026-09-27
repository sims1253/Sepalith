whitelist <- function(object, patterns) {
  from <- "whitelist"
  validate_class(object, "cbtf", from = from)
  validate_class(patterns, "character", from = from, remove_empty = TRUE)

  ## join all regular expression patterns
  joined_patterns <- paste0(patterns, collapse = "|")
  joined_patterns <- gsub("^\\|", "", joined_patterns) # remove extra |

  ## apply the new whitelist patterns to errors and warnings
  object$runs <- lapply(object$runs, function(x) {
    x$res[grepl(joined_patterns, x$msg) & x$msg != "SKIP"] <- "OK"
    x
  })
  object$ignore_patterns <- setdiff(c(object$ignore_patterns, patterns), "")
  object
}
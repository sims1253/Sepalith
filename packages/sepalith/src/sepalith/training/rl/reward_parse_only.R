#!/usr/bin/env Rscript
# Fixed syntax-only harness. It never source()s, eval()s, or executes input code.
#
# Exit codes (reward v2.1):
#   0  the single input parses, or batch/version mode finished
#   3  the single input is not valid R (the designated syntax-error code)
#   2  usage error
# Any other exit, a signal or a timeout is an infrastructure failure. Callers
# must retry or stop; they must never score it as a model failure.
#
#   reward_parse_only.R INPUT_R             single file, exit 0 or 3
#   reward_parse_only.R --batch LIST OUT    parse every path in LIST (one per
#                                           line); write "<index>\t<0|3>" per
#                                           input and a final "DONE\t<count>"
#   reward_parse_only.R --version           print the parser identity
syntax_ok <- function(path) {
  # A missing or unreadable input is an infrastructure failure, not a syntax
  # failure: stop() outside tryCatch exits with status 1.
  if (!file.exists(path) || file.access(path, 4L) != 0L) {
    stop("input is missing or unreadable: ", path, call. = FALSE)
  }
  tryCatch({
    parse(file = path, keep.source = FALSE, encoding = "UTF-8")
    TRUE
  }, error = function(condition) FALSE)
}
r_identity <- function() {
  sprintf("R %s.%s (%s-%s-%s)", R.version$major, R.version$minor,
          R.version$year, R.version$month, R.version$day)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1L && args[[1]] == "--version") {
  cat(r_identity(), "\n", sep = "")
  quit(status = 0L, save = "no")
}
if (length(args) == 3L && args[[1]] == "--batch") {
  paths <- readLines(args[[2]], encoding = "UTF-8", warn = FALSE)
  codes <- vapply(paths, function(path) if (syntax_ok(path)) 0L else 3L, integer(1),
                  USE.NAMES = FALSE)
  lines <- c(sprintf("%d\t%d", seq_along(codes) - 1L, codes),
             sprintf("DONE\t%d", length(codes)))
  writeLines(lines, args[[3]], useBytes = TRUE)
  quit(status = 0L, save = "no")
}
if (length(args) != 1L || startsWith(args[[1]], "--")) {
  cat("usage: reward_parse_only.R INPUT_R | --batch LIST OUT | --version\n", file = stderr())
  quit(status = 2L, save = "no")
}
quit(status = if (syntax_ok(args[[1]])) 0L else 3L, save = "no")

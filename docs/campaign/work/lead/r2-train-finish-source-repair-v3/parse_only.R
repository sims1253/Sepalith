args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: parse_only.R PATHS OUTPUT")
paths <- readLines(args[[1L]], warn = FALSE, encoding = "UTF-8")
out <- file(args[[2L]], open = "wt", encoding = "UTF-8")
on.exit(close(out), add = TRUE)
for (path in paths) {
  ok <- tryCatch({ parse(file = path, keep.source = FALSE); TRUE }, error = function(e) FALSE)
  writeLines(if (ok) "1" else "0", out, sep = "\n", useBytes = TRUE)
}

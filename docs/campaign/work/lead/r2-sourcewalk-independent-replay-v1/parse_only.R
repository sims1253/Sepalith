args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: parse_only.R input.tsv output.tsv")
items <- readLines(args[[1L]], warn = FALSE)
out <- file(args[[2L]], open = "wt")
on.exit(close(out), add = TRUE)
writeLines(paste("R_VERSION", R.version.string, sep = "\t"), out)
for (item in items) {
  fields <- strsplit(item, "\t", fixed = TRUE)[[1L]]
  if (length(fields) != 2L) stop("invalid input row")
  status <- "parse_ok"
  error_class <- ""
  tryCatch(
    parse(file = fields[[2L]], keep.source = FALSE),
    error = function(e) {
      status <<- "parse_error"
      error_class <<- class(e)[[1L]]
    }
  )
  writeLines(paste(fields[[1L]], status, error_class, sep = "\t"), out)
}

# Parse data of R files for the mutant generator (never evaluates code).
# Usage: Rscript --vanilla r_tokens.R OUT.tsv FILE...
# Columns: file index (0-based), id, parent, token, terminal, line1, col1, line2, col2, text.
args <- commandArgs(trailingOnly = TRUE)
out <- file(args[[1]], open = "w", encoding = "UTF-8")
for (i in seq_along(args)[-1]) {
  exprs <- tryCatch(parse(args[[i]], keep.source = TRUE), error = function(e) NULL)
  if (is.null(exprs)) {
    writeLines(paste(i - 2L, "ERROR", sep = "\t"), out)
    next
  }
  pd <- getParseData(exprs, includeText = TRUE)
  pd <- pd[order(pd$line1, pd$col1, -pd$line2, -pd$col2), ]
  text <- gsub("[\t\r\n]", " ", ifelse(pd$terminal, pd$text, ""))
  writeLines(paste(i - 2L, pd$id, pd$parent, pd$token, as.integer(pd$terminal), pd$line1, pd$col1,
                   pd$line2, pd$col2, text, sep = "\t"), out, useBytes = TRUE)
}
close(out)

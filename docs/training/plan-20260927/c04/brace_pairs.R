# C04 helper (same parse logic as C06's check_parse.R): parse documents and list brace pairs.
#
# Usage: Rscript --vanilla brace_pairs.R <manifest.tsv> <results.tsv>
#
# manifest.tsv has a header and columns key, path. For each file the result
# records whether it parses and, if it does, every `{`/`}` pair as
# "open_line:close_line:owner_line" (1-based). The owner line is where the
# construct owning the braced block starts (for a function: its header line).
args <- commandArgs(trailingOnly = TRUE)
manifest <- utils::read.delim(args[[1]], stringsAsFactors = FALSE, quote = "")
out <- character(nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  path <- manifest$path[[i]]
  text <- readChar(path, file.info(path)$size, useBytes = TRUE)
  Encoding(text) <- "UTF-8"
  parsed <- tryCatch(
    parse(text = text, keep.source = TRUE, encoding = "UTF-8"),
    error = function(e) e
  )
  ok <- !inherits(parsed, "error")
  pairs <- ""
  err <- ""
  if (!ok) {
    err <- gsub("[\t\r\n]+", " ", conditionMessage(parsed))
  } else {
    pd <- utils::getParseData(parsed, includeText = FALSE)
    closes <- pd[pd$token == "'}'", c("line1", "parent")]
    opens <- pd[pd$token == "'{'", c("line1", "parent")]
    m <- match(closes$parent, opens$parent)
    braced <- match(closes$parent, pd$id)
    owner <- match(pd$parent[braced], pd$id)
    owner_line <- ifelse(is.na(owner), opens$line1[m], pd$line1[owner])
    pairs <- paste(paste0(opens$line1[m], ":", closes$line1, ":", owner_line), collapse = ",")
  }
  out[[i]] <- paste(manifest$key[[i]], as.integer(ok), pairs, substr(err, 1, 200), sep = "\t")
}
writeLines(c("key\tparse_ok\tbrace_pairs\terror", out), args[[2]], useBytes = TRUE)

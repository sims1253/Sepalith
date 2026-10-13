# C06 helper: parse checks for reconstructed finish_block documents.
#
# Usage: Rscript --vanilla check_parse.R <manifest.tsv> <results.tsv>
#
# manifest.tsv has a header and columns: key, kind, path. `kind` is one of
# `applied` (gold applied document), `pre` (pre-edit document) or `nobrace`
# (applied document without the target's final `}`). For every row the result
# records whether the file parses. For `applied` rows that parse it also lists
# every `{`/`}` pair as "open_line:close_line" so the caller can check that the
# final brace closes a block opened in the prompt and compare indentation.
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
  } else if (manifest$kind[[i]] == "applied") {
    pd <- utils::getParseData(parsed, includeText = FALSE)
    closes <- pd[pd$token == "'}'", c("line1", "parent")]
    opens <- pd[pd$token == "'{'", c("line1", "parent")]
    m <- match(closes$parent, opens$parent)
    # Line where the construct owning the block starts (function, if, for,
    # call argument ...): the parent of the braced expression.
    braced <- match(closes$parent, pd$id)
    owner <- match(pd$parent[braced], pd$id)
    owner_line <- ifelse(is.na(owner), opens$line1[m], pd$line1[owner])
    pairs <- paste(paste0(opens$line1[m], ":", closes$line1, ":", owner_line), collapse = ",")
  }
  out[[i]] <- paste(manifest$key[[i]], manifest$kind[[i]], as.integer(ok), pairs, substr(err, 1, 200), sep = "\t")
}
writeLines(c("key\tkind\tparse_ok\tbrace_pairs\terror", out), args[[2]], useBytes = TRUE)

# C10 helper: parse checks for reconstructed roxygen documents.
#
# Usage: Rscript --vanilla check_roxygen.R <manifest.tsv> <results.tsv>
#
# manifest.tsv has a header and columns: key, path. Each file is the applied
# document: selected prefix spans, the target roxygen block, then the code
# after it. For every file the result records whether the whole document
# parses, whether the text from the first roxygen line onward parses, and the
# first top-level expression after the roxygen block: its object name, kind
# (function, or the call head such as setMethod or R6Class) and, for
# functions, the formal argument names.
args <- commandArgs(trailingOnly = TRUE)
manifest <- utils::read.delim(args[[1]], stringsAsFactors = FALSE, quote = "")
clean <- function(x) gsub("[\t\r\n]+", " ", x)
try_parse <- function(text) {
  tryCatch(parse(text = text, keep.source = TRUE, encoding = "UTF-8"), error = function(e) e)
}
out <- character(nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  path <- manifest$path[[i]]
  text <- readChar(path, file.info(path)$size, useBytes = TRUE)
  Encoding(text) <- "UTF-8"
  lines <- strsplit(text, "\n", fixed = TRUE)[[1]]
  rox <- which(grepl("^\\s*#'", lines))
  first <- if (length(rox)) rox[[1]] else 1L
  # End of the first contiguous roxygen block.
  last <- first
  while (last + 1L <= length(lines) && grepl("^\\s*#'", lines[[last + 1L]])) last <- last + 1L
  doc <- try_parse(text)
  tail <- try_parse(paste(lines[first:length(lines)], collapse = "\n"))
  object <- "NA"
  kind <- "NA"
  formals <- "NA"
  err <- ""
  if (inherits(doc, "error")) err <- clean(conditionMessage(doc))
  if (!inherits(tail, "error")) {
    refs <- attr(tail, "srcref")
    starts <- vapply(refs, function(r) r[[1]], integer(1)) + first - 1L
    j <- which(starts > last)
    if (length(j)) {
      e <- tail[[j[[1]]]]
      if (is.call(e) && as.character(e[[1]])[[1]] %in% c("<-", "=", "<<-") && length(e) == 3L) {
        object <- paste(deparse(e[[2]]), collapse = "")
        rhs <- e[[3]]
        if (is.call(rhs) && identical(rhs[[1]], as.name("function"))) {
          kind <- "function"
          formals <- paste(names(rhs[[2]]), collapse = ",")
        } else if (is.call(rhs)) {
          kind <- paste(deparse(rhs[[1]]), collapse = "")
        } else {
          kind <- class(rhs)[[1]]
        }
      } else if (is.call(e)) {
        kind <- paste(deparse(e[[1]]), collapse = "")
        if (length(e) > 1L && is.character(e[[2]])) object <- e[[2]]
      } else {
        kind <- class(e)[[1]]
        object <- paste(deparse(e), collapse = "")
      }
    }
  } else if (!nzchar(err)) {
    err <- clean(conditionMessage(tail))
  }
  out[[i]] <- paste(manifest$key[[i]], as.integer(!inherits(doc, "error")), as.integer(!inherits(tail, "error")),
                    clean(object), clean(kind), formals, substr(err, 1, 200), sep = "\t")
}
writeLines(c("key\tdoc_parse_ok\ttail_parse_ok\tobject\tkind\tformals\terror", out), args[[2]], useBytes = TRUE)

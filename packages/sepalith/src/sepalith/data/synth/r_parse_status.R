# Parse-only status for each file argument: ok, incomplete (unexpected end of
# input) or error. Never evaluates code. Output: "<index>\t<status>" per file.
args <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(args)) {
  status <- tryCatch({
    invisible(parse(args[[i]], keep.source = FALSE))
    "ok"
  }, error = function(e) {
    if (grepl("unexpected end of input", conditionMessage(e), fixed = TRUE)) "incomplete" else "error"
  })
  cat(i - 1L, "\t", status, "\n", sep = "")
}

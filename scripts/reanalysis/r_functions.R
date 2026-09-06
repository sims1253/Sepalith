# Parse source without evaluating it. Emit no source text or parser diagnostics.
for (path in commandArgs(trailingOnly = TRUE)) {
  expressions <- tryCatch(parse(file = path, keep.source = FALSE),
                          error = function(e) NULL)
  if (is.null(expressions)) {
    cat("ERROR\n")
    next
  }
  names <- character()
  for (expr in expressions) {
    if (is.call(expr) && length(expr) == 3L &&
        as.character(expr[[1L]])[1L] %in% c("<-", "=") &&
        is.symbol(expr[[2L]]) && is.call(expr[[3L]]) &&
        identical(expr[[3L]][[1L]], as.name("function"))) {
      name <- charToRaw(enc2utf8(as.character(expr[[2L]])))
      names <- c(names, paste(sprintf("%02x", as.integer(name)), collapse = ""))
    }
  }
  cat(paste(c("OK", length(expressions), names), collapse = "\t"), "\n", sep = "")
}

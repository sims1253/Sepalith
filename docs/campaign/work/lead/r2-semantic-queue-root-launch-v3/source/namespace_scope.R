#!/usr/bin/env Rscript
# Parse-only NAMESPACE inventory. Directives are inspected as AST nodes and
# are never evaluated; packages are never loaded.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('usage: namespace_scope.R INPUT.json OUTPUT.jsonl')
input <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
atom_name <- function(x) {
  if (is.symbol(x)) {
    value <- as.character(x)
    if (length(value)==1L && nzchar(value)) return(value)
  }
  if (is.character(x) && length(x)==1L && nzchar(x)) return(x)
  NULL
}
output <- file(args[[2]], 'wt')
on.exit(close(output), add=TRUE)
for (item in input$namespaces) {
  parsed_sha256 <- digest::digest(file=item$path, algo='sha256', serialize=FALSE)
  parsed <- tryCatch(parse(file=item$path, keep.source=TRUE, encoding='UTF-8'), error=identity)
  if (inherits(parsed, 'error')) {
    result <- list(id=item$id, status='hold_namespace_parse_error', parsed_sha256=parsed_sha256,
      imported_symbols=list(), error=conditionMessage(parsed))
    writeLines(jsonlite::toJSON(result, auto_unbox=TRUE, null='null'), output)
    next
  }
  imported <- character(); malformed <- character()
  for (node in as.list(parsed)) {
    if (!is.call(node) || !is.symbol(node[[1]]) || !identical(as.character(node[[1]]), 'importFrom')) next
    if (length(node) < 3L) {
      malformed <- c(malformed, deparse(node, width.cutoff=500L))
      next
    }
    values <- character(); valid <- TRUE
    for (index in 2:length(node)) {
      value <- tryCatch(atom_name(node[[index]]), error=function(e) NULL)
      if (is.null(value)) { valid <- FALSE; break }
      values <- c(values, value)
    }
    if (!valid || length(values) < 2L) malformed <- c(malformed, deparse(node, width.cutoff=500L))
    else imported <- c(imported, values[-1L])
  }
  result <- if (length(malformed))
    list(id=item$id, status='hold_namespace_malformed_importFrom', parsed_sha256=parsed_sha256,
      imported_symbols=as.list(sort(unique(imported))), malformed=as.list(malformed), error=NULL)
  else list(id=item$id, status='namespace_inventory_complete', parsed_sha256=parsed_sha256,
      imported_symbols=as.list(sort(unique(imported))), malformed=list(), error=NULL)
  writeLines(jsonlite::toJSON(result, auto_unbox=TRUE, null='null'), output)
}

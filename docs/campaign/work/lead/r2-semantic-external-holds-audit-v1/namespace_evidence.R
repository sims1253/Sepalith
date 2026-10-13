#!/usr/bin/env Rscript
# Parse-only NAMESPACE evidence. Directives are inspected as AST and never evaluated.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('usage: namespace_evidence.R INPUT.json OUTPUT.jsonl')
input <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
atom <- function(x) {
  if (is.symbol(x) || (is.character(x) && length(x)==1L)) {
    value <- as.character(x)
    if (length(value)==1L && nzchar(value)) return(value)
  }
  NULL
}
output <- file(args[[2]], 'wt'); on.exit(close(output), add=TRUE)
for (item in input$namespaces) {
  before <- file.info(item$path)
  parsed_sha256 <- digest::digest(file=item$path, algo='sha256', serialize=FALSE)
  parsed <- tryCatch(parse(file=item$path, keep.source=TRUE, encoding='UTF-8'), error=identity)
  after <- file.info(item$path)
  stable <- identical(before$size,after$size) && identical(as.numeric(before$mtime),as.numeric(after$mtime))
  if (inherits(parsed,'error')) {
    result <- list(id=item$id,status='hold_namespace_parse_error',parsed_sha256=parsed_sha256,stat_stable=stable,directives=list(),malformed=list(),error=conditionMessage(parsed))
    writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null'),output); next
  }
  directives <- list(); malformed <- list()
  for (node in as.list(parsed)) {
    if (!is.call(node) || !is.symbol(node[[1]])) next
    kind <- as.character(node[[1]])
    if (!(kind %in% c('importFrom','import'))) next
    values <- list(); valid <- TRUE
    if (length(node)>=2L) for (index in 2:length(node)) {
      value <- tryCatch(atom(node[[index]]),error=function(e) NULL)
      if (is.null(value)) { valid <- FALSE; break }
      values[[length(values)+1L]] <- value
    }
    need <- if (identical(kind,'importFrom')) 2L else 1L
    if (!valid || length(values)<need) malformed[[length(malformed)+1L]] <- paste(deparse(node,width.cutoff=500L),collapse=' ')
    else if (identical(kind,'importFrom')) directives[[length(directives)+1L]] <- list(kind=kind,package=values[[1L]],symbols=values[-1L])
    else for (package in values) directives[[length(directives)+1L]] <- list(kind=kind,package=package,symbols=list())
  }
  status <- if (!stable) 'hold_namespace_unstable' else if (length(malformed)) 'hold_namespace_malformed' else 'namespace_evidence_complete'
  result <- list(id=item$id,status=status,parsed_sha256=parsed_sha256,stat_stable=stable,directives=directives,malformed=malformed,error=NULL)
  writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null'),output)
}

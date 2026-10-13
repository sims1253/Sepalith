#!/usr/bin/env Rscript
# Read one bounded byte snapshot, hash those exact bytes, then parse only that
# text. NAMESPACE expressions are inspected as AST and never evaluated.
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
walk_imports <- function(node, depth=0L) {
  out <- list()
  if (!is.call(node)) return(out)
  head <- if (is.symbol(node[[1L]])) as.character(node[[1L]]) else ''
  if (head %in% c('importFrom','import')) out[[length(out)+1L]] <- list(node=node,depth=depth)
  if (length(node)>=2L) for (index in 2:length(node)) out <- c(out,walk_imports(node[[index]],depth+1L))
  out
}
output <- file(args[[2]], 'wt'); on.exit(close(output), add=TRUE)
for (item in input$namespaces) {
  max_bytes <- if (is.null(item$max_bytes)) 1048576L else as.integer(item$max_bytes)
  before <- file.info(item$path)
  if (is.na(before$size) || before$size < 0 || before$size > max_bytes) {
    result <- list(id=item$id,status='hold_namespace_size',parsed_sha256=NULL,stat_stable=FALSE,directives=list(),malformed=list(),conditional=list(),error='missing or exceeds bounded size')
    writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null'),output); next
  }
  con <- file(item$path,'rb'); bytes <- readBin(con,'raw',n=as.integer(before$size)); close(con)
  parsed_sha256 <- digest::digest(bytes,algo='sha256',serialize=FALSE)
  text <- tryCatch(rawToChar(bytes),error=identity)
  parse_eol <- NULL
  if (!inherits(text,'error')) {
    has_crlf <- grepl('\r\n',text,fixed=TRUE)
    without_crlf <- gsub('\r\n','',text,fixed=TRUE)
    has_lone_cr <- grepl('\r',without_crlf,fixed=TRUE)
    has_lf <- grepl('\n',without_crlf,fixed=TRUE)
    parse_eol <- if (has_lone_cr || (has_crlf && has_lf)) 'unsupported_mixed_or_lone_cr' else if (has_crlf) 'uniform_crlf_to_lf' else 'lf'
    if (identical(parse_eol,'uniform_crlf_to_lf')) text <- gsub('\r\n','\n',text,fixed=TRUE)
  }
  parsed <- if (inherits(text,'error')) text else if (identical(parse_eol,'unsupported_mixed_or_lone_cr')) simpleError(parse_eol) else tryCatch(parse(text=text,keep.source=TRUE,encoding='UTF-8'),error=identity)
  after <- file.info(item$path)
  stable <- identical(before$size,after$size) && identical(as.numeric(before$mtime),as.numeric(after$mtime)) && length(bytes)==before$size
  if (!is.null(item$expected_sha256) && !identical(parsed_sha256,item$expected_sha256)) stable <- FALSE
  if (inherits(parsed,'error')) {
    result <- list(id=item$id,status='hold_namespace_parse_error',parsed_sha256=parsed_sha256,parse_eol=parse_eol,stat_stable=stable,directives=list(),malformed=list(),conditional=list(),error=conditionMessage(parsed))
    writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null'),output); next
  }
  directives <- list(); malformed <- list(); conditional <- list()
  for (top in as.list(parsed)) {
    found <- walk_imports(top,0L)
    for (entry in found) {
      node <- entry$node
      kind <- as.character(node[[1L]])
      if (entry$depth != 0L) {
        conditional[[length(conditional)+1L]] <- paste(deparse(node,width.cutoff=500L),collapse=' ')
        next
      }
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
  }
  status <- if (!stable) 'hold_namespace_unstable' else if (length(conditional)) 'hold_namespace_conditional_directives' else if (length(malformed)) 'hold_namespace_malformed' else 'namespace_evidence_complete'
  result <- list(id=item$id,status=status,parsed_sha256=parsed_sha256,parse_eol=parse_eol,stat_stable=stable,directives=directives,malformed=malformed,conditional=conditional,error=NULL)
  writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null'),output)
}

#!/usr/bin/env Rscript
# Parse-only semantic inventory. Function bodies and generated targets are never executed.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop('usage: semantic_scope.R INPUT.json OUTPUT.jsonl')
input <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
head_name <- function(x) if (is.call(x) && is.symbol(x[[1]])) as.character(x[[1]]) else NA_character_
find_definitions <- function(exprs) {
 out <- list()
 for (node in as.list(exprs)) {
  if (!is.call(node) || !(head_name(node) %in% c('<-','=','<<-')) || length(node)!=3L || !is.symbol(node[[2]])) next
  name <- as.character(node[[2]]); value <- node[[3]]; ref <- attr(node,'srcref')
  out[[length(out)+1L]] <- list(name=name, is_function=is.call(value)&&identical(head_name(value),'function'),
    start_line=if(is.null(ref)) NA_integer_ else as.integer(ref[[1]]), end_line=if(is.null(ref)) NA_integer_ else as.integer(ref[[3]]), value=value)
 }
 out
}
safe_globals <- function(fun) tryCatch({
 g <- codetools::findGlobals(eval(fun,envir=baseenv()),merge=FALSE)
 list(ok=TRUE, variables=sort(unique(as.character(g$variables))), functions=sort(unique(as.character(g$functions))), error=NULL)
}, error=function(e) list(ok=FALSE,variables=character(),functions=character(),error=conditionMessage(e)))
output <- file(args[[2]],'wt');on.exit(close(output),add=TRUE)
for (group in input$source_groups) {
 parsed <- tryCatch(parse(file=group$source_path,keep.source=TRUE,encoding='UTF-8'),error=identity)
 if (inherits(parsed,'error')) {
  for(row in group$rows) writeLines(jsonlite::toJSON(list(row_id=row$row_id,status='hold_source_parse_error',error=conditionMessage(parsed)),auto_unbox=TRUE,null='null'),output)
  next
 }
 defs <- find_definitions(parsed); top_names <- vapply(defs,function(x)x$name,character(1))
 for(row in group$rows) {
  hits <- Filter(function(x)isTRUE(x$is_function)&&identical(x$name,row$target_definition_name),defs)
  if(length(hits)!=1L) result <- list(row_id=row$row_id,status='hold_target_definition_not_unique',target_matches=length(hits))
  else {
   hit <- hits[[1]]; fun <- hit$value; globals <- safe_globals(fun); formal_names <- names(fun[[2]]); if(is.null(formal_names))formal_names<-character()
   base_names <- unique(c(globals$variables,globals$functions));base_bound <- sort(base_names[vapply(base_names,function(n)exists(n,envir=baseenv(),inherits=FALSE),logical(1))])
   definition_spans <- setNames(lapply(defs,function(x)list(x$start_line,x$end_line)),top_names)
   definition_counts <- as.list(table(top_names))
   result <- list(row_id=row$row_id,status=if(globals$ok)'scope_inventory_complete' else 'hold_codetools_failure',target_definition_name=row$target_definition_name,
    target_definition_span=list(hit$start_line,hit$end_line),function_formals=as.list(formal_names),codetools_ok=globals$ok,
    variable_references=as.list(globals$variables),call_heads=as.list(globals$functions),base_bound=as.list(base_bound),top_level_definitions=as.list(sort(unique(top_names))),top_level_definition_spans=definition_spans,top_level_definition_counts=definition_counts,codetools_error=globals$error)
  }
  writeLines(jsonlite::toJSON(result,auto_unbox=TRUE,null='null',na='null',digits=NA),output)
 }
}

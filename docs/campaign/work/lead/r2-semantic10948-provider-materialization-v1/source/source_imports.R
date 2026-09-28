#!/usr/bin/env Rscript
# Infer import-relevant names from one pre-edit TRAIN/editor snapshot. Function
# bodies are inspected by codetools but never called; package code is not loaded.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L)stop('usage: source_imports.R INPUT.json OUTPUT.json')
x <- jsonlite::fromJSON(args[[1]],simplifyVector=FALSE)
raw <- jsonlite::base64_dec(x$source_base64)
if(length(raw)>x$max_bytes)stop('source snapshot exceeds bound')
actual <- digest::digest(raw,algo='sha256',serialize=FALSE)
if(!identical(actual,x$source_sha256))stop('source snapshot hash differs')
text <- rawToChar(raw)
has_crlf <- grepl('\r\n',text,fixed=TRUE);without <- gsub('\r\n','',text,fixed=TRUE)
if(grepl('\r',without,fixed=TRUE)||(has_crlf&&grepl('\n',without,fixed=TRUE)))stop('mixed or lone CR')
if(has_crlf)text<-gsub('\r\n','\n',text,fixed=TRUE)
parsed<-parse(text=text,keep.source=TRUE,encoding='UTF-8');refs<-attr(parsed,'srcref');nodes<-as.list(parsed)
headname<-function(v)if(is.call(v)&&is.symbol(v[[1L]]))as.character(v[[1L]])else''
defs<-list()
for(i in seq_along(nodes)){n<-nodes[[i]];if(!is.call(n)||!(headname(n)%in%c('<-','=','<<-'))||length(n)!=3L||!is.symbol(n[[2L]])||!is.call(n[[3L]])||headname(n[[3L]])!='function')next;r<-attr(n,'srcref');if(is.null(r)&&length(refs)>=i)r<-refs[[i]];if(is.null(r))next;defs[[length(defs)+1L]]<-list(name=as.character(n[[2L]]),fun=n[[3L]],start=as.integer(r[[1L]]),end=as.integer(r[[3L]]))}
following<-Filter(function(d)d$start>as.integer(x$cursor_line)+1L,defs);if(!length(following))stop('following function absent');starts<-vapply(following,function(d)d$start,integer(1));target<-following[[which.min(starts)]]
byname<-split(defs,vapply(defs,function(d)d$name,character(1)));seen<-character();external<-character();queue<-list(target)
available<-unique(unlist(x$explicit_import_symbols,use.names=FALSE))
while(length(queue)){d<-queue[[1L]];queue<-queue[-1L];if(d$name%in%seen)next;seen<-c(seen,d$name);g<-codetools::findGlobals(eval(d$fun,envir=baseenv()),merge=FALSE);names<-sort(unique(c(as.character(g$variables),as.character(g$functions))));for(n in names){if(n%in%available){external<-c(external,n);next};if(exists(n,envir=baseenv(),inherits=FALSE))next;if(!is.null(byname[[n]])&&length(byname[[n]])==1L)queue[[length(queue)+1L]]<-byname[[n]][[1L]]else external<-c(external,n)}}
used<-sort(unique(intersect(external,available)))
jsonlite::write_json(list(status='source_import_inventory_complete',source_sha256=actual,target_name=target$name,target_span=list(target$start,target$end),import_dependencies=as.list(used),unresolved_nonimport=as.list(sort(unique(setdiff(external,available))))),args[[2]],auto_unbox=TRUE,pretty=FALSE)

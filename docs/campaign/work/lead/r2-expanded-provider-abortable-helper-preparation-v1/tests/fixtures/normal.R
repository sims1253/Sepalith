args <- commandArgs(trailingOnly = TRUE)
writeLines(as.character(Sys.getpid()), args[[1]])
writeLines('{"status":"ok"}', args[[2]])

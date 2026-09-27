args <- commandArgs(trailingOnly = TRUE)
writeLines(as.character(Sys.getpid()), args[[1]])
Sys.sleep(5)
writeLines('unexpected', args[[2]])

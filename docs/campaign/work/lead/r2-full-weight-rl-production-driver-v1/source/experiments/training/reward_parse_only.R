#!/usr/bin/env Rscript
# Fixed syntax-only harness. It never source()s, eval()s, or executes input code.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: reward_parse_only.R INPUT_R", call. = FALSE)
status <- tryCatch({
  parse(file = args[[1]], keep.source = FALSE, encoding = "UTF-8")
  TRUE
}, error = function(condition) FALSE)
quit(status = if (status) 0L else 1L, save = "no")

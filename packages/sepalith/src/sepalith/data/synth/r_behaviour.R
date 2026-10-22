# Behavioural check: source a seed file into a fresh environment, then evaluate
# its check file there. Base R and recommended packages only.
# Usage: Rscript --vanilla r_behaviour.R SEED.R CHECK.R  -> prints "ok" or "fail\t<message>".
args <- commandArgs(trailingOnly = TRUE)
env <- new.env(parent = globalenv())
status <- tryCatch({
  sys.source(args[[1]], envir = env)
  sys.source(args[[2]], envir = env)
  "ok"
}, error = function(e) paste0("fail\t", gsub("[\r\n\t]+", " ", conditionMessage(e))))
cat(status, "\n", sep = "")

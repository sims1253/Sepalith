# Behavioural check for one seed file (base R and recommended packages only).
# Usage: Rscript --vanilla r_behaviour.R SEED.R CHECK.R
# Output, two lines:
#   status\t<ok | fail\t<message>>    plain run: seed sourced, check evaluated (decides pass/fail)
#   entered\t<name,name,...>          traced run: closures defined by the seed whose body was
#                                     entered while the check ran (trace() on every closure, so
#                                     indirect calls and S3 dispatch through these bindings count)
# "Entered" means the function body started executing; it does not mean the check asserted
# anything about that function's result.
args <- commandArgs(trailingOnly = TRUE)
clean <- function(x) gsub("[\r\n\t]+", " ", x)

plain <- new.env(parent = globalenv())
status <- tryCatch({
  sys.source(args[[1]], envir = plain)
  sys.source(args[[2]], envir = plain)
  "ok"
}, error = function(e) paste0("fail\t", clean(conditionMessage(e))))
cat("status\t", status, "\n", sep = "")

entered <- new.env()
traced <- new.env(parent = globalenv())
invisible(tryCatch({
  sys.source(args[[1]], envir = traced)
  for (name in ls(traced, all.names = TRUE)) {
    f <- get(name, envir = traced)
    if (is.function(f) && !is.primitive(f)) {
      tracer <- substitute(assign(N, TRUE, envir = E), list(N = name, E = entered))
      suppressMessages(trace(name, tracer = tracer, where = traced, print = FALSE))
    }
  }
  sys.source(args[[2]], envir = traced)
}, error = function(e) NULL))
cat("entered\t", paste(sort(ls(entered, all.names = TRUE)), collapse = ","), "\n", sep = "")

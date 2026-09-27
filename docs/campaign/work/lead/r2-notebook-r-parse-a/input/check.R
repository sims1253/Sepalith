files <- sort(list.files("documents", pattern="[.]R$", full.names=TRUE))
out <- lapply(files, function(f) {
  result <- tryCatch({parse(file=f, keep.source=FALSE); list(ok=TRUE, error="")}, error=function(e) list(ok=FALSE, error=conditionMessage(e)))
  data.frame(path=f, ok=result$ok, error=result$error, stringsAsFactors=FALSE)
})
write.csv(do.call(rbind,out), "r-parse-results.csv", row.names=FALSE)
cat(R.version.string, "\n")
cat("parsed", length(files), "failures", sum(!vapply(out,function(x) x$ok,logical(1))), "\n")

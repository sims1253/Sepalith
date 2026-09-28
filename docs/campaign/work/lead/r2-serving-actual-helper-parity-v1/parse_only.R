args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
fixture <- jsonlite::fromJSON(args[[1]], simplifyVector=FALSE)
parse_one <- function(text) {
  tryCatch({ parse(text=text, keep.source=TRUE); list(status="parse_ok", error=NULL) },
           error=function(e) list(status="syntax_error", error=conditionMessage(e)))
}
result <- list(schema="sepalith.run06.actual_helper_parse_only.v1",
               baseline=parse_one(fixture$preedit_text),
               reapplied=parse_one(fixture$source_after_text),
               generated_r_executed=FALSE,
               r_version=R.version.string)
writeLines(jsonlite::toJSON(result, auto_unbox=TRUE, pretty=TRUE), args[[2]])
cat("PARSE_ONLY_COMPLETE\n")

#!/usr/bin/env Rscript
# Parse-only harness. It never source()s, eval()s, or executes an input buffer.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("usage: parse_only.R INPUT_TSV OUTPUT_TSV", call. = FALSE)
input <- read.delim(args[[1]], header = TRUE, sep = "\t", quote = "", comment.char = "",
                    colClasses = "character", check.names = FALSE)
required <- c("kind", "arm", "case_index", "id", "family", "path", "sha256")
if (!identical(names(input), required)) stop("input columns differ", call. = FALSE)
parse_one <- function(path) {
  started <- proc.time()[["elapsed"]]
  result <- tryCatch({
    parsed <- parse(file = path, keep.source = FALSE, encoding = "UTF-8")
    list(ok = TRUE, expressions = length(parsed), error = "")
  }, error = function(condition) {
    list(ok = FALSE, expressions = NA_integer_, error = conditionMessage(condition))
  })
  result$elapsed_ms <- (proc.time()[["elapsed"]] - started) * 1000
  result
}
rows <- lapply(seq_len(nrow(input)), function(index) {
  value <- parse_one(input$path[[index]])
  data.frame(kind = input$kind[[index]], arm = input$arm[[index]],
             case_index = input$case_index[[index]], id = input$id[[index]],
             family = input$family[[index]], sha256 = input$sha256[[index]],
             parse_ok = if (value$ok) "true" else "false",
             expressions = if (is.na(value$expressions)) "" else as.character(value$expressions),
             elapsed_ms = sprintf("%.3f", value$elapsed_ms), error = value$error,
             stringsAsFactors = FALSE, check.names = FALSE)
})
output <- do.call(rbind, rows)
write.table(output, file = args[[2]], sep = "\t", quote = TRUE, row.names = FALSE,
            col.names = TRUE, na = "")

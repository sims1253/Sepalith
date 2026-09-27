.select_report_columns <- function(tab, cols) {
  cols <- intersect(cols, names(tab))
  out <- tab[, cols, drop = FALSE]
  for (col in names(out)) {
    if (is.numeric(out[[col]])) out[[col]] <- .fmt_num(out[[col]])
  }
  out
}
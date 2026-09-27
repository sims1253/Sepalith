.bind_rows <- function(rows) {
  rows <- Filter(Negate(is.null), rows)
  if (length(rows) == 0) {
    return(data.frame())
  }
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out
}
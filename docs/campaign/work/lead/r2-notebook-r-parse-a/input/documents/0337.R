ensure_proper_table <- function(x) {
  if (is.null(names(dimnames(x)))) {
    names(dimnames(x)) <- ""
  }
  x
}
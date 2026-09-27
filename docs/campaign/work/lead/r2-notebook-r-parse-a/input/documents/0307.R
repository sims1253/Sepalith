extract_matrix <- function(x, i, j) {
  subset <- .Call(C_BEDMatrix_extract_matrix, slot(x, "xptr"), i, j)
  # Preserve dimnames
  names <- slot(x, "dnames")
  dimnames(subset) <- list(
    names[[1L]][i],
    names[[2L]][j]
  )
  return(subset)
}
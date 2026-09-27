convert_named_numeric_to_table <- function(named_numeric_vector) {
  output <- as.integer(named_numeric_vector)
  names(output) <- names(named_numeric_vector)
  output <- as.table(output)
  names(dimnames(output)) <- ""
  output
}
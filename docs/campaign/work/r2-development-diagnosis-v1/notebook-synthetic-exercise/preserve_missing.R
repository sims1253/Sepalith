# Return NA when any input is missing; do not ignore missing values.
sum_preserving_missing <- function(x) {
  sum(x, na.rm = FALSE)
}

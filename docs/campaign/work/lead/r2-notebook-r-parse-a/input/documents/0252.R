compute_matrix <- function(intersections_as_groups, sorted_groups) {
  matrix <- sapply(
    intersections_as_groups,
    function(i_groups) {
      sorted_groups %in% i_groups
    },
    simplify = FALSE
  )

  matrix_data <- as.data.frame(
    matrix,
    row.names = sorted_groups,
    check.names = FALSE
  )
  matrix_data
}
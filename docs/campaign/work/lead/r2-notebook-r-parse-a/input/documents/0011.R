double_ml_data_from_data_frame <- function(
  df,
  x_cols = NULL,
  y_col = NULL,
  d_cols = NULL,
  z_cols = NULL,
  s_col = NULL,
  cluster_cols = NULL,
  use_other_treat_as_covariate = TRUE
) {
  if (is.null(cluster_cols)) {
    data <- DoubleMLData$new(
      df,
      x_cols = x_cols,
      y_col = y_col,
      d_cols = d_cols,
      z_cols = z_cols,
      s_col = s_col,
      use_other_treat_as_covariate = use_other_treat_as_covariate
    )
  } else {
    data <- DoubleMLClusterData$new(
      df,
      x_cols = x_cols,
      y_col = y_col,
      d_cols = d_cols,
      z_cols = z_cols,
      s_col = s_col,
      cluster_cols = cluster_cols,
      use_other_treat_as_covariate = use_other_treat_as_covariate
    )
  }
  return(data)
}
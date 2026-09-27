double_ml_data_from_matrix <- function(
  X = NULL,
  y,
  d,
  z = NULL,
  s = NULL,
  cluster_vars = NULL,
  data_class = "DoubleMLData",
  use_other_treat_as_covariate = TRUE
) {
  assert_choice(
    data_class,
    c(
      "DoubleMLData",
      "data.table",
      "DoubleMLClusterData"
    )
  )
  assert_logical(use_other_treat_as_covariate, len = 1)

  y <- assure_matrix(y)
  d <- assure_matrix(d)
  mat_list <- list(y, d)

  if (!is.null(X)) {
    X <- assure_matrix(X)
    mat_list[[length(mat_list) + 1]] <- X
  }
  if (!is.null(z)) {
    z <- assure_matrix(z)
    mat_list[[length(mat_list) + 1]] <- z
  }
  if (!is.null(s)) {
    s <- assure_matrix(s)
    mat_list[[length(mat_list) + 1]] <- s
  }
  if (!is.null(cluster_vars)) {
    cluster_vars <- assure_matrix(cluster_vars)
    mat_list[[length(mat_list) + 1]] <- cluster_vars
  }

  check_matrix_row(mat_list)
  data <- data.table(X, y, d, z, s, cluster_vars)

  if (!is.null(z)) {
    if (ncol(z) == 1) {
      z_cols <- "z"
    } else {
      z_cols <- paste0("z", seq_len(ncol(z)))
    }
  } else {
    z_cols <- NULL
  }
  y_col <- "y"
  if (ncol(d) == 1) {
    d_cols <- "d"
  } else {
    d_cols <- paste0("d", seq_len(ncol(d)))
  }
  if (!is.null(X)) {
    x_cols <- paste0("X", seq_len(ncol(X)))
  } else {
    x_cols <- NULL
  }
  if (!is.null(s)) {
    s_col <- "s"
  } else {
    s_col <- NULL
  }
  if (!is.null(cluster_vars)) {
    if (ncol(cluster_vars) == 1) {
      cluster_cols <- "cluster_var"
    } else {
      cluster_cols <- paste0("cluster_var", seq_len(ncol(z)))
    }
  } else {
    cluster_cols <- NULL
  }
  names(data) <- c(x_cols, y_col, d_cols, z_cols, s_col, cluster_cols)

  if (data_class %in% c("DoubleMLData", "DoubleMLClusterData")) {
    if (is.null(cluster_vars)) {
      if (data_class == "DoubleMLClusterData") {
        stop(paste(
          "To initialize a DoubleMLClusterData object a matrix of cluster",
          "variables (`cluster_vars`) must be provided."
        ))
      }
      data <- DoubleMLData$new(
        data,
        x_cols = x_cols,
        y_col = y_col,
        d_cols = d_cols,
        z_cols = z_cols,
        s_col = s_col,
        use_other_treat_as_covariate = use_other_treat_as_covariate
      )
    } else {
      data <- DoubleMLClusterData$new(
        data,
        x_cols = x_cols,
        y_col = y_col,
        d_cols = d_cols,
        z_cols = z_cols,
        s_col = s_col,
        cluster_cols = cluster_cols,
        use_other_treat_as_covariate = use_other_treat_as_covariate
      )
    }
  }
  return(data)
}
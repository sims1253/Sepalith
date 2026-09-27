formatCovariateSummary_internal <- function(x, metric, catEncoding = NULL,
                                            coh = TRUE) {
  counts <- countCovariateTypes_internal(x, metric)
  parts <- character(0)
  
  if (counts$continuous > 0) {
    parts <- c(parts, sprintf("%d continuous", counts$continuous))
  }
  if (counts$spherical > 0) {
    parts <- c(parts, sprintf("%d spherical", counts$spherical))
  }
  if (counts$categorical > 0) {
    parts <- c(parts, sprintf("%d categorical", counts$categorical))
  }
  
  if (length(parts) == 0) {
    return(character(0))
  }
  
  lines <- c(sprintf("Covariate summary: %s.", paste(parts, collapse = ", ")))
  
  if (counts$categorical > 0 && coh) {
    if (!is.null(catEncoding) && !is.null(catEncoding$encodedBinaryCols)) {
      binary_cols <- length(catEncoding$encodedBinaryCols)
      lines <- c(
        lines,
        sprintf(
          paste0(
            "Categorical covariates are expanded to %d one-hot encoded ",
            "binary column%s, with the first level of each categorical ",
            "variable used as the reference category."
          ),
          binary_cols,
          if (binary_cols == 1L) "" else "s"
        )
      )
    } else {
      lines <- c(
        lines,
        paste0(
          "Categorical covariates are expanded to one-hot encoded binary ",
          "columns, with the first level of each categorical variable used ",
          "as the reference category."
        )
      )
    }
  }
  
  lines
}
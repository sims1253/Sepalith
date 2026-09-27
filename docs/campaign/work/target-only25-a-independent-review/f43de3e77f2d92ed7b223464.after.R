#' Automatic Stratification
#'
#' Automatically partitions a population into homogeneous strata using
#' clustering-based methods. The resulting strata can be used in survey
#' sampling designs to improve the precision and efficiency of estimates.
#'
#' @param data A data frame containing the study variables.
#' @param target A target variable used for stratification.
#' @param n_strata An integer specifying the desired number of strata.
#'
#' @return
#' A data frame containing the original data along with the assigned
#' stratum membership. The returned object has class `"autostrata"`.
#'
#' @examples
#' data <- data.frame(
#'   x = c(10, 15, 20, 25, 30, 35),
#'   y = c(5, 8, 12, 16, 20, 24)
#' )
#'
#' result <- autostrata(
#'   data = data,
#'   target = y,
#'   n_strata = 2
#' )
#'
#' head(result)
#'
#' @export
autostrata <- function(data, target, n_strata = 4) {
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame.")
  }
  if (!is.numeric(n_strata) || length(n_strata) != 1) {
    stop("'n_strata' must be a single positive integer.")
  }
  if (n_strata < 1) {
    stop("'n_strata' must be greater than 0.")
  }
  if (!target %in% colnames(data)) {
    stop("'target' must be a column name in 'data'.")
  }

  # Preprocessing
  x <- data[[target]]
  x <- x[!is.na(x)]
  if (length(x) == 0) {
    stop("No valid non-missing values found in the target variable.")
  }

  # Clustering
  cl <- hclust(dist(x), "ward.D")
  clusters <- cutree(cl, n_strata)

  # Assign strata
  result <- data
  result$stratum <- as.character(clusters)

  return(result)
}
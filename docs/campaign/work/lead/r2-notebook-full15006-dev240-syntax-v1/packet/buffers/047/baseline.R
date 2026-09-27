#' Evaluate Strata
#'
#' Computes measures of stratification quality, including the overall
#' variance of the target variable, the total within-stratum variance,
#' and a homogeneity index.
#'
#' @param data A stratified data frame containing a column named
#'   `stratum`.
#' @param target Character string specifying the target variable used
#'   for stratification.
#'
#' @return
#' A list containing:
#' \describe{
#'   \item{overall_variance}{Variance of the target variable in the full population.}
#'   \item{within_variance}{Sum of within-stratum variances.}
#'   \item{homogeneity}{Homogeneity index, where larger values indicate
#'   more homogeneous strata.}
#' }
#'
#' @examples
#' data <- data.frame(
#'   income = c(15, 18, 20, 25, 30, 35, 40, 50),
#'   stratum = c(1, 1, 2, 2, 3, 3, 4, 4)
#' )
#'
#' evaluate_strata(
#'   data = data,
#'   target = "income"
#' )
#'
#' @export
evaluate_strata <- function(data, target) {
  overall_var <- stats::var(data[[target]], na.rm = TRUE)

  within_var <- dplyr::summarise(
    dplyr::group_by(data, stratum),
    v = var(.data[[target]])
  )

  homogeneity <- 1 -
    sum(within_var$v) / overall_var

  list(
    overall_variance = overall_var,
    within_variance = sum(within_var$v),
    homogeneity = homogeneity
  )
}

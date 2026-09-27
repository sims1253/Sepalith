#' Compare Efficiency
#'
#' Compares variance under simple random sampling (SRS)
#' with within-stratum variance after stratification.
#'
#' @param data A stratified data frame containing a variable named
#'   \code{stratum}.
#' @param target Character string giving the target variable name.
#'
#' @return A numeric value representing relative efficiency.
#' Values greater than 1 indicate improved efficiency due to stratification.
#'
#' @importFrom stats var
#' @importFrom stats na.omit
#' @importFrom dplyr group_by summarise
#' @importFrom rlang .data
#'
#' @export
compare_sampling <- function(data, target) {
  srs_var <- stats::var(data[[target]], na.rm = TRUE)

  strat_var <- data |>
    dplyr::group_by(stratum) |>
    dplyr::summarise(
      v = stats::var(.data[[target]], na.rm = TRUE),
      .groups = "drop"
    )

  srs_var / mean(strat_var$v, na.rm = TRUE)
}

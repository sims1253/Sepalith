causal_estimand_inference <- function(
  outcome_model_results,
  df,
  continuous_name,
  categorical_name,
  treatment_name,
  treatment_value,
  previous_status_name,
  credible_interval_level = 0.95,
  num_of_grids = 50
) {
  # Replaced parallel processing with a fast closed-form standard lapply
  causal_estimand_output <- lapply(
    X = seq(0, 1, length.out = num_of_grids),
    FUN = function(x) {
      causal_estimand_single_value(
        outcome_model_results = outcome_model_results,
        df = df,
        continuous_name = continuous_name,
        categorical_name = categorical_name,
        treatment_name = treatment_name,
        treatment_value = treatment_value,
        previous_status_name = previous_status_name,
        previous_status_value = x,
        credible_interval_level = credible_interval_level
      )
    }
  )

  return(causal_estimand_output)
}
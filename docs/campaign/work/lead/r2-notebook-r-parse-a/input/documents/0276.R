causal_estimand_single_value <- function(
  outcome_model_results,
  df,
  continuous_name,
  categorical_name,
  treatment_name,
  treatment_value,
  previous_status_name,
  previous_status_value,
  credible_interval_level = 0.95
) {
  num_samples <- nrow(outcome_model_results$fx_output)
  num_warmup <- outcome_model_results$num_warmup
  thin <- outcome_model_results$thin
  model_type <- outcome_model_results$model_type

  # if df is a tibble, convert it as dataframe
  df <- as.data.frame(df)
  # define X_continuous
  X_continuous <- as.data.frame(df[continuous_name])
  # given previous status value
  X_continuous[, previous_status_name] <- previous_status_value
  X_continuous <- as.matrix(X_continuous)
  # define X_categorical
  X_categorical <- as.data.frame(df[categorical_name])
  # given treatment type
  X_categorical[, treatment_name] <- treatment_value

  # create dataset for prediction
  if (model_type %in% c("Tobit-XBART", "N-XBART")) {
    # convert to numeric
    X_categorical <- sapply(X_categorical, as.numeric)
    Xall <- cbind(X_continuous, X_categorical)
    forest_dataset <- createForestDataset(Xall)
  } else if (model_type %in% c("Tobit-SBART", "N-SBART")) {
    X_categorical <- model.matrix(~ . - 1, data = X_categorical)
    Xall <- cbind(X_continuous, X_categorical)
  } else if (model_type %in% c("Tobit-LH", "N-LH")) {
    Xall <- generate_design_matrix(
      X_continuous = as.data.frame(X_continuous),
      X_categorical = X_categorical
    )
  }

  # Initialize the causal estimand output utilizing closed forms
  causal_estimand_output <- lapply(1:num_samples, function(i) {
    # Calculate the correct index in the full, unthinned MCMC chain
    actual_idx <- num_warmup + (i * thin)

    if (model_type %in% c("Tobit-XBART", "N-XBART")) {
      fx_value <- outcome_model_results$forest_samples$predict_raw_single_forest(
        forest_dataset = forest_dataset,
        forest_num = actual_idx - 1
      )
    } else if (model_type %in% c("Tobit-SBART", "N-SBART")) {
      fx_value <- outcome_model_results$softforest$predict_iteration(
        X = Xall,
        i = actual_idx - 1
      )
    } else if (model_type %in% c("Tobit-LH", "N-LH")) {
      fx_value <- Xall %*% outcome_model_results$beta_output[i, ]
    }
    fx_value <- as.numeric(fx_value)

    global_error_variance_value <- outcome_model_results$global_error_variance_output[
      i
    ]
    random_effects_variance_value <- outcome_model_results$random_effects_variance_output[
      i
    ]

    if (model_type %in% c("Tobit-XBART", "Tobit-SBART", "Tobit-LH")) {
      # Closed-form parameters
      sigma_sq <- global_error_variance_value
      sigma_star <- sqrt(sigma_sq + random_effects_variance_value)

      # Convolution terms for MAPO and MPDR
      term_1 <- pnorm((1 - fx_value) / sigma_star)
      term_2 <- pnorm(-fx_value / sigma_star)
      density_1 <- dnorm(-fx_value / sigma_star)
      density_2 <- dnorm((1 - fx_value) / sigma_star)

      # MAPO (Continuous Expectation): G-computation via mean() over all observations
      mapo_values <- fx_value *
        (term_1 - term_2) +
        sigma_star * (density_1 - density_2) +
        (1 - term_1)
      mapo_value <- mean(mapo_values)

      # MPDR (Probability of Disease Resolution): G-computation via mean() over all observations
      mpdr_values <- term_2
      mpdr_value <- mean(mpdr_values)
    } else if (model_type %in% c("N-XBART", "N-SBART", "N-LH")) {
      # for normal likelihood, the E(y) = f(X). Just take average of f(X).
      mapo_value <- mean(fx_value)
      mpdr_value <- NA # MPDR is typically defined via the Tobit boundary mass
    }

    output <- list(
      mapo_value = mapo_value,
      mpdr_value = mpdr_value,
      fx_value = fx_value
    )

    return(output)
  })

  # Convert the list to numeric vectors
  mapo_value <- sapply(causal_estimand_output, function(x) {
    x$mapo_value
  })
  mpdr_value <- sapply(causal_estimand_output, function(x) {
    x$mpdr_value
  })
  fx_value <- do.call(
    "rbind",
    lapply(causal_estimand_output, function(x) {
      x$fx_value
    })
  )

  # Summarize MAPO
  df_mapo_output <- data.frame(
    treatment = treatment_value,
    previous_status = previous_status_value,
    mapo_mean = mean(mapo_value),
    mapo_lb = quantile(mapo_value, probs = (1 - credible_interval_level) / 2),
    mapo_ub = quantile(
      mapo_value,
      probs = 1 - (1 - credible_interval_level) / 2
    )
  )
  row.names(df_mapo_output) <- NULL

  # Summarize MPDR
  df_mpdr_output <- data.frame(
    treatment = treatment_value,
    previous_status = previous_status_value,
    mpdr_mean = mean(mpdr_value, na.rm = TRUE),
    mpdr_lb = quantile(
      mpdr_value,
      probs = (1 - credible_interval_level) / 2,
      na.rm = TRUE
    ),
    mpdr_ub = quantile(
      mpdr_value,
      probs = 1 - (1 - credible_interval_level) / 2,
      na.rm = TRUE
    )
  )
  row.names(df_mpdr_output) <- NULL

  output <- list(
    treatment = treatment_value,
    previous_status = previous_status_value,
    credible_interval_level = credible_interval_level,
    mapo_summary = df_mapo_output,
    mpdr_summary = df_mpdr_output,
    mapo_value = mapo_value,
    mpdr_value = mpdr_value,
    fx_value = fx_value
  )

  return(output)
}
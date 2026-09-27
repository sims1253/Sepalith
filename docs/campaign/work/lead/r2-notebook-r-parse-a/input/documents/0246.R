simulate_dewpoint <- function(
  temperature,
  rh,
  min_dewpoint = -5,
  max_dewpoint = 35,
  noise_sd = 0.5,
  seed = NULL
) {
  # -----------------------------------
  # Validation
  # -----------------------------------
  if (missing(temperature)) {
    stop(
      "temperature data is required. ",
      "Use simulate_temperature() first."
    )
  }
  if (missing(rh)) {
    stop(
      "Relative humidity data is required. ",
      "Use simulate_rh() first."
    )
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
  required_temp <- c(
    "Station",
    "DATE",
    "Avg.Temp"
  )
  required_rh <- c(
    "Station",
    "DATE",
    "RH"
  )
  if (!all(required_temp %in% names(temperature))) {
    stop(
      "temperature must contain: ",
      paste(required_temp, collapse = ", ")
    )
  }
  if (!all(required_rh %in% names(rh))) {
    stop(
      "rh must contain: ",
      paste(required_rh, collapse = ", ")
    )
  }
  # -----------------------------------
  # Merge temperature + RH
  # -----------------------------------
  sim_df <- merge(
    temperature,
    rh[, c("Station", "DATE", "RH")],
    by = c("Station", "DATE")
  )
  # -----------------------------------
  # Magnus dew point equation
  # -----------------------------------
  a <- 17.27
  b <- 237.7
  gamma_val <-
    (a * sim_df$Avg.Temp / (b + sim_df$Avg.Temp)) + log(sim_df$RH / 100)
  sim_df$DewPoint <- (b * gamma_val) / (a - gamma_val)
  # -----------------------------------
  # Add stochastic variability
  # -----------------------------------
  sim_df$DewPoint <-
    sim_df$DewPoint +
    rnorm(
      nrow(sim_df),
      mean = 0,
      sd = noise_sd
    )
  # -----------------------------------
  # Physical constraints
  # -----------------------------------
  # Dew point cannot exceed air temperature
  sim_df$DewPoint <- pmin(sim_df$DewPoint, sim_df$Avg.Temp)
  # Apply bounds
  sim_df$DewPoint <-
    pmax(
      min_dewpoint,
      pmin(
        max_dewpoint,
        sim_df$DewPoint
      )
    )

  # -----------------------------------
  # Derived metrics
  # -----------------------------------
  sim_df$Dewpoint_Depression <- sim_df$Avg.Temp - sim_df$DewPoint
  # -----------------------------------
  # Round outputs
  # -----------------------------------
  numeric_cols <- c(
    "Avg.Temp",
    "RH",
    "DewPoint",
    "Dewpoint_Depression"
  )
  sim_df[numeric_cols] <- round(sim_df[numeric_cols], 2)
  # -----------------------------------
  # Final output
  # -----------------------------------
  keep_cols <- c(
    "Station",
    "LON",
    "LAT",
    "ELEV",
    "DATE",
    "Year",
    "Month",
    "Season",
    "Avg.Temp",
    "RH",
    "DewPoint",
    "Dewpoint_Depression"
  )
  sim_df <- sim_df[, keep_cols]
  message(
    "Dew point simulation complete for ",
    length(unique(sim_df$Station)),
    " stations."
  )
  return(sim_df)
}
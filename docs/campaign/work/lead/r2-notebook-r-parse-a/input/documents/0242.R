generate_horizontal_layout <- function(
  NewBook,
  plots,
  n_TrtGen,
  n_Reps,
  planter
) {
  layouts <- list()

  # B. Basic horizontal grid
  basic_horizontal_df <- NewBook |>
    dplyr::mutate(
      ROW = rep(1:n_TrtGen, times = n_Reps),
      COLUMN = rep(1:n_Reps, each = n_TrtGen)
    )
  layouts[["basic_horizontal"]] <- basic_horizontal_df

  # C. Extended factor-based layouts
  factor_combos <- as.data.frame(
    factor_subsets(n_TrtGen, all_factors = TRUE)$comb_factors
  )
  if (nrow(factor_combos) > 0) {
    for (i in seq_len(nrow(factor_combos))) {
      s1 <- as.numeric(factor_combos[i, 1])
      s2 <- as.numeric(factor_combos[i, 2])

      # Assign columns per replication
      w <- 1:(s1 * n_Reps)
      u <- seq(1, length(w), by = s1)
      v <- seq(s1, length(w), by = s1)
      z <- unlist(lapply(1:n_Reps, function(j) rep(u[j]:v[j], times = s2)))

      df <- NewBook |>
        dplyr::mutate(
          ROW = rep(rep(1:s2, each = s1), n_Reps),
          COLUMN = z
        )
      nCols <- max(df$COLUMN)

      df$PLOT <- planter_transform(
        plots = plots,
        planter = planter,
        reps = n_Reps,
        cols = nCols,
        mode = "Horizontal",
        units = NULL
      )
      layouts[[paste0("horizontal_ext_", i)]] <- df
    }
  }

  return(layouts)
}
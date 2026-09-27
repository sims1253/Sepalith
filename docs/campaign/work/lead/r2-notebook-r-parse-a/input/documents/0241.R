generate_vertical_layout <- function(
  NewBook,
  plots,
  n_TrtGen,
  n_Reps,
  planter
) {
  layouts <- list()

  # 1) Basic vertical layout: (n_Reps) rows, (n_TrtGen) columns
  basic_df <- NewBook |>
    dplyr::mutate(
      ROW = rep(1:n_Reps, each = n_TrtGen),
      COLUMN = rep(1:n_TrtGen, times = n_Reps)
    )
  layouts[["basic_vertical"]] <- basic_df

  # 2) Extended vertical factor-based layouts
  pf <- numbers::primeFactors(n_TrtGen)
  if (length(pf) >= 2) {
    factor_combos <- as.data.frame(
      factor_subsets(n_TrtGen, all_factors = TRUE)$comb_factors
    )
    for (i in seq_len(nrow(factor_combos))) {
      s1 <- as.numeric(factor_combos[i, 1])
      s2 <- as.numeric(factor_combos[i, 2])

      df <- NewBook |>
        dplyr::mutate(
          ROW = rep(1:(s1 * n_Reps), each = s2),
          COLUMN = rep(rep(1:s2, times = s1), times = n_Reps)
        )
      nCols <- max(df$COLUMN)

      df$PLOT <- planter_transform(
        plots = plots,
        planter = planter,
        reps = n_Reps,
        cols = nCols,
        units = NULL
      )
      layouts[[paste0("vertical_ext_", i)]] <- df
    }
  }

  # 3) Single-column layout (all plots in a single column) as the LAST option
  single_column_df <- NewBook |>
    dplyr::mutate(
      ROW = 1:(n_TrtGen * n_Reps),
      COLUMN = 1
    )
  layouts[["single_column"]] <- single_column_df

  return(layouts)
}
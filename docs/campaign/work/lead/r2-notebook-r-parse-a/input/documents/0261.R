plot_thermal_heatmap <- function(
  img_obj,
  use_processed = TRUE,
  palette = "inferno"
) {
  # Check object class
  if (!inherits(img_obj, "BioThermR")) {
    stop("Error: Input must be a 'BioThermR' object.")
  }

  mat <- if (use_processed) img_obj$processed else img_obj$raw

  if (is.null(mat)) {
    stop("Error: Selected matrix is empty.")
  }

  df <- expand.grid(
    row = seq_len(nrow(mat)),
    col = seq_len(ncol(mat))
  )
  df$val <- c(mat)

  p <- ggplot(df, aes(x = col, y = row, fill = val)) +
    geom_raster() +
    scale_fill_viridis_c(
      option = palette,
      name = "Temp (\u00B0C)",
      na.value = "transparent"
    ) +
    coord_fixed() +
    theme_void() +
    labs(
      title = paste("Thermal Heatmap:", img_obj$meta$filename),
      subtitle = paste(
        "Source:",
        if (use_processed) "Processed Data" else "Raw Data"
      )
    )

  return(p)
}
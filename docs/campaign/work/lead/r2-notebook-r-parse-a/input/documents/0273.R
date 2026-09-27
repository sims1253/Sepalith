plot_BRRAT_by_id <- function(
  fit_obj,
  n_grid = 150,
  level = 0.80,
  facet = TRUE
) {
  stopifnot(inherits(fit_obj, 'BRRAT'))

  id_levels <- fit_obj$id_levels
  J <- length(id_levels)

  # Extract per-ID generated quantities (iterations x J)
  intercept_draws <- rstan::extract(fit_obj$fit, pars = 'intercept_id')[[1]]
  slope_draws <- rstan::extract(fit_obj$fit, pars = 'slope_id')[[1]]

  x_min <- min(fit_obj$data$x, na.rm = TRUE)
  x_max <- max(fit_obj$data$x, na.rm = TRUE)
  x_std <- seq(x_min, x_max, length.out = n_grid)

  q_lo <- (1 - level) / 2
  q_hi <- 1 - q_lo

  out_list <- vector('list', J)
  for (j in seq_len(J)) {
    mu_draws_j <- sapply(x_std, function(xi) {
      intercept_draws[, j] + slope_draws[, j] * xi
    })
    mu_mean <- colMeans(mu_draws_j)
    mu_lo <- apply(mu_draws_j, 2, stats::quantile, probs = q_lo)
    mu_hi <- apply(mu_draws_j, 2, stats::quantile, probs = q_hi)
    out_list[[j]] <- data.frame(
      ID = id_levels[j],
      P = x_std,
      mu_mean = mu_mean,
      mu_lo = mu_lo,
      mu_hi = mu_hi
    )
  }
  df_lines <- do.call(rbind, out_list)

  # Downsample points per ID if very large
  data_sc <- data.frame(
    x = fit_obj$data$x,
    y = fit_obj$data$y,
    ID = fit_obj$id_levels[fit_obj$data$group]
  )
  if (nrow(data_sc) > 30000) {
    data_sc <- do.call(
      rbind,
      lapply(split(data_sc, data_sc$ID), function(dd) {
        keep <- min(5000, nrow(dd))
        dd[sample(seq_len(nrow(dd)), keep), ]
      })
    )
  }

  if (!requireNamespace('ggplot2', quietly = TRUE)) {
    stop('ggplot2 is required.')
  }

  p <- ggplot2::ggplot() +
    ggplot2::geom_point(
      data = data_sc,
      ggplot2::aes(x = .data$x, y = .data$y, colour = .data$ID),
      alpha = 0.12,
      size = 0.6
    ) +
    ggplot2::geom_ribbon(
      data = df_lines,
      ggplot2::aes(
        x = .data$P,
        ymin = .data$mu_lo,
        ymax = .data$mu_hi,
        fill = .data$ID
      ),
      alpha = 0.15
    ) +
    ggplot2::geom_line(
      data = df_lines,
      ggplot2::aes(x = .data$P, y = .data$mu_mean, colour = .data$ID),
      linewidth = 1
    ) +
    ggplot2::labs(
      x = 'P (normalized)',
      y = 'Qerr (transformed)',
      title = sprintf(
        'Per-site partial-pooling lines (%.0f%% credible bands)',
        level * 100
      )
    ) +
    ggplot2::theme_minimal()

  if (facet) {
    p <- p +
      ggplot2::facet_wrap(~ID, scales = 'free_y', ncol = 3) +
      ggplot2::guides(colour = 'none', fill = 'none')
  }
  return(p)
}
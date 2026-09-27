estimate_sigma_equivalent <- function(
  real_data,
  its = 10,
  nin = 2,
  nsp = 12,
  c_opt = NULL,
  c_synth = NULL,
  sigmas,
  iter_max = 100,
  make_plot = TRUE,
  return_plot = TRUE,
  quiet = FALSE
) {
  stopifnot(is.matrix(real_data) || is.data.frame(real_data))
  X <- as.matrix(real_data)
  X <- X[complete.cases(X), , drop = FALSE]

  n <- nrow(X)
  p <- ncol(X)

  # 1. Real AMD curve and c_opt
  if (is.null(c_opt)) {
    res_real <- compute_amd_curve(
      data = X,
      its = its,
      nin = nin,
      nsp = nsp,
      iter_max = iter_max,
      verbose = !quiet
    )
    c_opt <- res_real$c_opt
  } else {
    res_real <- compute_amd_curve(
      data = X,
      its = its,
      nin = c_opt,
      nsp = c_opt,
      iter_max = iter_max,
      verbose = !quiet
    )
  }

  amd_real_peak <- max(res_real$max, na.rm = TRUE)

  # Number of synthetic configurations
  if (is.null(c_synth)) {
    c_synth <- c_opt
  }

  # 2. Synthetic \u03C3\u2013AMDmax sweep
  L <- 100
  amd_synth <- numeric(length(sigmas))

  base_seed <- 123456L

  for (i in seq_along(sigmas)) {
    sigma <- sigmas[i]
    if (!quiet) {
      cat("Sigma =", sigma, "\n")
    }

    # 2.1 Centroids in hypercube interior
    centers <- matrix(
      runif(c_synth * p, min = 0.2 * L, max = 0.8 * L),
      nrow = c_synth,
      ncol = p
    )

    # 2.2 Dimension-wise heterogeneity
    scaling_factors <- runif(p, min = 0.5, max = 1.5)

    # 2.3 Points per cluster
    points_per_cluster <- floor(n / c_synth)

    # 2.4 Generate synthetic dataset
    synth <- do.call(
      rbind,
      lapply(1:c_synth, function(k) {
        noise <- matrix(
          rnorm(points_per_cluster * p, mean = 0, sd = sigma * scaling_factors),
          nrow = points_per_cluster,
          ncol = p,
          byrow = TRUE
        )
        noise +
          matrix(
            centers[k, ],
            nrow = points_per_cluster,
            ncol = p,
            byrow = TRUE
          )
      })
    )

    # 2.5 Synthetic AMD peak
    res_synth <- compute_amd_curve(
      data = synth,
      its = 1,
      nin = c_synth,
      nsp = c_synth,
      iter_max = iter_max,
      verbose = FALSE
    )

    amd_synth[i] <- max(res_synth$max, na.rm = TRUE)
  }

  # 3. Interpolation
  sigma_eq <- NA_real_
  extrapolated <- FALSE

  if (amd_real_peak <= min(amd_synth)) {
    sigma_eq <- min(sigmas)
    extrapolated <- TRUE
  } else if (amd_real_peak >= max(amd_synth)) {
    sigma_eq <- max(sigmas)
    extrapolated <- TRUE
  } else {
    sigma_eq <- approx(
      x = amd_synth,
      y = sigmas,
      xout = amd_real_peak
    )$y
  }

  # 4. Output table
  table_sigma_amd <- data.frame(
    sigma = sigmas,
    AMD_max = amd_synth
  )

  # 5. Plot
  plot_obj <- NULL
  if (make_plot) {
    plot(
      sigmas,
      amd_synth,
      type = "b",
      pch = 16,
      xlab = "Sigma",
      ylab = "Synthetic AMD_max",
      main = "Sigma\u2013AMDmax calibration"
    )
    abline(h = amd_real_peak, col = "red", lty = 2)
    abline(v = sigma_eq, col = "blue", lty = 3)
  }

  if (return_plot) {
    plot_obj <- list(
      sigmas = sigmas,
      amd_synth = amd_synth,
      amd_real_peak = amd_real_peak,
      sigma_eq = sigma_eq
    )
  }

  # 6. Final output
  out <- list(
    c_opt = c_opt,
    amd_real_peak = amd_real_peak,
    sigma_eq = sigma_eq,
    sigma_equivalent = sigma_eq,
    extrapolated = extrapolated,
    table_sigma_amd = table_sigma_amd,
    plot_comparative = plot_obj
  )

  class(out) <- "sigma_equivalent"
  return(out)
}
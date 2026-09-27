char_lowess <- function(y, x = NULL, span = 0.1, iter = 0L) {
  n <- length(y)
  if (is.null(x)) {
    x <- seq_len(n)
  }

  # ---- NaN handling ----------------------------------------------------------
  # MATLAB's smooth() excludes NaN positions from local WLS fits and from the
  # bisquare residual median, then returns NA at the NaN centre positions.
  # The previous approach bridged NaN in CharSmooth before calling this
  # function, but that changes the bisquare weight trajectory on subsequent
  # passes, causing charBkg to diverge for records with gaps when using
  # method 2 (robust lowess).  Handling NaN here directly matches smooth().
  nan_mask <- is.na(y)
  has_nan <- any(nan_mask)

  # ---- Convert span to integer window width ---------------------------------
  # Mirrors charLowess.m lines 63-68:
  #   if span < 1:  k = max(3, round(span * n))   % fraction -> point count
  #   else:         k = max(3, round(span))        % already a point count
  if (span < 1) {
    k <- max(3L, round(span * n))
  } else {
    k <- max(3L, round(span))
  }
  k <- min(k, n)
  hw <- k %/% 2L # floor(k/2): nominal half-window

  n_iter <- 1L + as.integer(iter) # total passes: 1 plain, 5 robust

  ys <- y
  # NaN positions permanently get weight 0 so they never contribute to a fit.
  rw <- ifelse(nan_mask, 0.0, 1.0)

  for (it in seq_len(n_iter)) {
    for (i in seq_len(n)) {
      # ---- NaN centre: pass through as NA ------------------------------------
      if (nan_mask[i]) {
        ys[i] <- NA_real_
        next
      }

      # ---- Shifted window: always k points (charLowess.m lines 110-120) ----
      i0 <- i - hw
      i1 <- i0 + k - 1L
      if (i0 < 1L) {
        i0 <- 1L
        i1 <- k
      }
      if (i1 > n) {
        i1 <- n
        i0 <- n - k + 1L
      }

      xi <- x[i0:i1]
      yi <- y[i0:i1]
      ri <- rw[i0:i1]

      # ---- Tricubic weights (charLowess.m lines 128-130) --------------------
      # Normalised by distance to the FURTHEST VALID point in the window.
      # NaN positions are excluded from dmax so they don't artificially
      # compress the weights of valid neighbours, matching smooth()'s
      # behaviour of treating NaN points as absent from the window.
      nan_in_win <- is.na(yi)
      d <- abs(xi - x[i])
      d_valid <- d[!nan_in_win]
      if (length(d_valid) == 0L) {
        ys[i] <- NA_real_
        next
      }
      dmax <- max(d_valid) + .Machine$double.eps
      tri <- (1.0 - (d / dmax)^3)^3
      w <- tri * ri # NaN positions already have ri = 0

      # ---- Weighted least-squares linear fit (charLowess.m lines 133-147) --
      sw <- sum(w)
      if (sw < .Machine$double.eps) {
        ys[i] <- NA_real_
        next
      }

      # Replace NaN yi with 0 for arithmetic (weight is already 0 so they
      # contribute nothing to the sums, but NA would propagate otherwise).
      yi_safe <- yi
      yi_safe[nan_in_win] <- 0.0

      swx <- sum(w * xi)
      swy <- sum(w * yi_safe)
      swx2 <- sum(w * xi^2)
      swxy <- sum(w * xi * yi_safe)
      det <- sw * swx2 - swx^2

      if (abs(det) < .Machine$double.eps * max(abs(c(sw * swx2, swx^2)))) {
        ys[i] <- swy / sw # near-singular: mean
      } else {
        a <- (swy * swx2 - swx * swxy) / det
        b <- (sw * swxy - swx * swy) / det
        ys[i] <- a + b * x[i]
      }
    }

    # ---- Bisquare robustness weights (charLowess.m lines 151-157) ----------
    # Compute the scale (median absolute residual) only from valid positions,
    # matching MATLAB smooth() which skips NaN when computing the median.
    if (iter > 0L && it < n_iter) {
      res <- y - ys
      s_val <- stats::median(abs(res[!nan_mask]), na.rm = TRUE)
      if (is.na(s_val) || s_val < .Machine$double.eps) {
        break
      }
      u <- res / (6.0 * s_val)
      new_rw <- pmax(0.0, 1.0 - u^2)^2 * (abs(u) < 1.0)
      new_rw[nan_mask] <- 0.0 # keep NaN positions excluded
      rw <- new_rw
    }
  }

  ys
}
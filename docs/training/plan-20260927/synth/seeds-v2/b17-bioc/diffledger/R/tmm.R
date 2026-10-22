#' @title Pick the TMM reference sample
#' @description Chooses the sample whose upper-quartile scaled count is
#'   closest to the mean upper quartile across samples.
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lib_size Library sizes, one per sample.
#' @return Integer index of the reference column.
#' @noRd
.tmm_reference <- function(counts, lib_size) {
  upper_q <- numeric(ncol(counts))
  for (j in seq_len(ncol(counts))) {
    upper_q[j] <- stats::quantile(counts[, j] / lib_size[j], 0.75, names = FALSE)
  }

  # Very sparse data: every upper quartile is zero, so fall back to the
  # sample with the most (square-root damped) signal.
  if (all(upper_q == 0)) {
    return(unname(which.max(colSums(sqrt(counts)))))
  }
  which.min(abs(upper_q - mean(upper_q)))
}

#' @title Log-ratio (M) values between a sample and the reference
#' @param obs,ref Count vectors for the sample and the reference.
#' @param n_obs,n_ref Library sizes of the sample and the reference.
#' @noRd
.tmm_m_values <- function(obs, ref, n_obs, n_ref) {
  log2(obs / n_obs) - log2(ref / n_ref)
}

#' @title Absolute expression (A) values between a sample and the reference
#' @param obs,ref Count vectors for the sample and the reference.
#' @param n_obs,n_ref Library sizes of the sample and the reference.
#' @noRd
.tmm_a_values <- function(obs, ref, n_obs, n_ref) {
  (log2(obs / n_obs) + log2(ref / n_ref)) / 2
}

#' @title Asymptotic variance weights for M values
#' @param obs,ref Count vectors for the sample and the reference.
#' @param n_obs,n_ref Library sizes of the sample and the reference.
#' @noRd
.tmm_variance <- function(obs, ref, n_obs, n_ref) {
  (n_obs - obs) / n_obs / obs + (n_ref - ref) / n_ref / ref
}

#' @title Trimmed mean of M values for one sample
#' @description Trims features by both M and A value and returns the
#'   precision-weighted mean of the remaining log ratios as a linear scaling
#'   factor.
#' @param obs,ref Count vectors for the sample and the reference.
#' @param n_obs,n_ref Library sizes of the sample and the reference.
#' @param trim_m Fraction trimmed from each end of the M values.
#' @param trim_a Fraction trimmed from each end of the A values.
#' @param a_cutoff Features with A values at or below this are ignored.
#' @return A single positive scaling factor.
#' @noRd
.tmm_one <- function(obs, ref, n_obs, n_ref, trim_m = 0.3, trim_a = 0.05,
                     a_cutoff = -1e10) {
  m <- .tmm_m_values(obs, ref, n_obs, n_ref)
  a <- .tmm_a_values(obs, ref, n_obs, n_ref)
  v <- .tmm_variance(obs, ref, n_obs, n_ref)

  # Features with a zero in either sample give infinite M or A values.
  usable <- is.finite(m) & is.finite(a) & a > a_cutoff
  m <- m[usable]
  a <- a[usable]
  v <- v[usable]
  if (length(m) == 0L || max(abs(m)) < 1e-6) {
    return(1)
  }

  m_bounds <- stats::quantile(m, c(trim_m, 1 - trim_m), names = FALSE)
  a_bounds <- stats::quantile(a, c(trim_a, 1 - trim_a), names = FALSE)
  keep <- m >= m_bounds[1L] & m <= m_bounds[2L] &
    a >= a_bounds[1L] & a <= a_bounds[2L]
  if (!any(keep)) {
    return(1)
  }

  # Inverse-variance weighting: high-count features are more precise.
  2^(sum(m[keep] / v[keep]) / sum(1 / v[keep]))
}

#' @title TMM normalisation factors
#' @description Computes trimmed-mean-of-M-values normalisation factors for
#'   every sample against a reference, rescaled so that their product is one.
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lib_size Optional library sizes; defaults to column sums.
#' @param ref_column Optional index or name of the reference sample; chosen
#'   by upper quartile when `NULL`.
#' @param trim_m Fraction trimmed from each end of the M values.
#' @param trim_a Fraction trimmed from each end of the A values.
#' @return A named numeric vector of normalisation factors, one per sample.
#' @export
tmm_factors <- function(counts, lib_size = NULL, ref_column = NULL,
                        trim_m = 0.3, trim_a = 0.05) {
  check_counts(counts)
  if (is.null(lib_size)) {
    lib_size <- colSums(counts)
  }
  if (any(lib_size == 0)) {
    stop("all samples must have a positive library size", call. = FALSE)
  }

  # drop features that are zero everywhere; they carry no information
  counts <- counts[rowSums(counts) > 0, , drop = FALSE]
  if (is.null(ref_column)) {
    ref_column <- .tmm_reference(counts, lib_size)
  } else if (is.character(ref_column)) {
    ref_column <- match(ref_column, colnames(counts))
    if (is.na(ref_column)) {
      stop("`ref_column` does not name a sample in `counts`", call. = FALSE)
    }
  }

  ref <- counts[, ref_column]
  factors <- numeric(ncol(counts))
  for (j in seq_len(ncol(counts))) {
    if (j == ref_column) {
      factors[j] <- 1
    } else {
      factors[j] <- .tmm_one(
        counts[, j], ref,
        n_obs = lib_size[j],
        n_ref = lib_size[ref_column],
        trim_m = trim_m,
        trim_a = trim_a
      )
    }
  }

  factors <- factors / exp(mean(log(factors)))
  names(factors) <- colnames(counts)
  factors
}

#' @title Effective library sizes
#' @description Multiplies raw library sizes by normalisation factors.
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param norm_factors Optional normalisation factors, one per sample;
#'   computed with [tmm_factors()] when `NULL`.
#' @return A named numeric vector of effective library sizes.
#' @export
effective_lib_size <- function(counts, norm_factors = NULL) {
  if (is.null(norm_factors)) {
    norm_factors <- tmm_factors(counts)
  }
  lib_size <- colSums(counts)
  if (length(norm_factors) != length(lib_size)) {
    stop("`norm_factors` must have one value per sample", call. = FALSE)
  }

  out <- lib_size * norm_factors
  names(out) <- colnames(counts)
  out
}

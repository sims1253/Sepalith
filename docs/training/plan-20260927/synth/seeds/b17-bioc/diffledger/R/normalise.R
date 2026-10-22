#' Validate a count matrix
#'
#' Checks that counts are a non-negative numeric matrix without missing
#' values and with unique feature names.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @return `counts`, invisibly, when valid; otherwise an error is raised.
#' @keywords internal
check_counts <- function(counts) {
  if (!is.matrix(counts) || !is.numeric(counts)) {
    stop("`counts` must be a numeric matrix", call. = FALSE)
  }
  if (anyNA(counts)) {
    stop("`counts` must not contain missing values", call. = FALSE)
  }
  if (any(counts < 0)) {
    stop("`counts` must be non-negative", call. = FALSE)
  }
  if (!is.null(rownames(counts)) && anyDuplicated(rownames(counts)) > 0L) {
    stop("feature names in `counts` must be unique", call. = FALSE)
  }
  invisible(counts)
}

#' Counts per million
#'
#' Scales each sample to a library size of one million, optionally using
#' normalisation factors to form effective library sizes.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lib_size Optional library sizes; defaults to column sums.
#' @param norm_factors Optional normalisation factors, one per sample.
#' @return A numeric matrix of the same shape as `counts`.
#' @family normalisation
#' @export
cpm <- function(counts, lib_size = NULL, norm_factors = NULL) {
  check_counts(counts)
  if (is.null(lib_size)) {
    lib_size <- colSums(counts)
  }
  if (!is.null(norm_factors)) {
    lib_size <- lib_size * norm_factors
  }
  sweep(counts, 2L, lib_size / 1e6, "/")
}

#' Log counts per million
#'
#' Adds a prior count scaled to library size before taking log2, so that
#' zeros map to finite values.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lib_size Optional library sizes; defaults to column sums.
#' @param norm_factors Optional normalisation factors, one per sample.
#' @param prior_count Average count added to each observation.
#' @return A numeric matrix of the same shape as `counts`.
#' @family normalisation
#' @export
log_cpm <- function(counts, lib_size = NULL, norm_factors = NULL, prior_count = 2) {
  check_counts(counts)
  if (is.null(lib_size)) {
    lib_size <- colSums(counts)
  }
  if (!is.null(norm_factors)) {
    lib_size <- lib_size * norm_factors
  }
  prior <- prior_count * lib_size / mean(lib_size)
  adj_lib <- lib_size + 2 * prior
  log2(sweep(sweep(counts, 2L, prior, "+"), 2L, adj_lib / 1e6, "/"))
}

#' Reads per kilobase per million
#'
#' Divides counts per million by feature length in kilobases.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lengths Feature lengths in bases, one per row of `counts`.
#' @param norm_factors Optional normalisation factors, one per sample.
#' @return A numeric matrix of the same shape as `counts`.
#' @family normalisation
#' @export
rpkm <- function(counts, lengths, norm_factors = NULL) {
  check_counts(counts)
  if (length(lengths) != nrow(counts)) {
    stop("`lengths` must have one value per feature", call. = FALSE)
  }
  per_million <- cpm(counts, norm_factors = norm_factors)
  per_million / (lengths / 1e3)
}

#' Transcripts per million
#'
#' Normalises counts by feature length first and then rescales each sample
#' to sum to one million.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param lengths Feature lengths in bases, one per row of `counts`.
#' @return A numeric matrix of the same shape as `counts`.
#' @family normalisation
#' @export
tpm <- function(counts, lengths) {
  check_counts(counts)
  if (length(lengths) != nrow(counts)) {
    stop("`lengths` must have one value per feature", call. = FALSE)
  }
  rate <- counts / (lengths / 1e3)
  sweep(rate, 2L, colSums(rate) / 1e6, "/")
}

#' Median-of-ratios size factors
#'
#' Estimates per-sample size factors as the median ratio of each sample to
#' the geometric mean across samples, using features without zeros.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @return A named numeric vector of size factors, one per sample.
#' @family size factors
#' @export
size_factors_ratio <- function(counts) {
  check_counts(counts)
  log_counts <- log(counts)
  log_geo_mean <- rowMeans(log_counts)
  usable <- is.finite(log_geo_mean)
  if (!any(usable)) {
    stop("every feature has at least one zero; cannot compute ratios", call. = FALSE)
  }
  factors <- numeric(ncol(counts))
  for (j in seq_len(ncol(counts))) {
    ratios <- log_counts[usable, j] - log_geo_mean[usable]
    factors[j] <- exp(stats::median(ratios))
  }
  names(factors) <- colnames(counts)
  factors
}

#' Upper-quartile size factors
#'
#' Scales samples by the chosen quantile of their non-zero counts, then
#' centres the factors to a geometric mean of one.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param p Quantile of the non-zero counts to use.
#' @return A named numeric vector of size factors, one per sample.
#' @family size factors
#' @export
size_factors_quantile <- function(counts, p = 0.75) {
  check_counts(counts)
  expressed <- rowSums(counts) > 0
  factors <- numeric(ncol(counts))
  for (j in seq_len(ncol(counts))) {
    vals <- counts[expressed, j]
    factors[j] <- stats::quantile(vals[vals > 0], probs = p, names = FALSE)
  }
  factors <- factors / exp(mean(log(factors)))
  names(factors) <- colnames(counts)
  factors
}

#' Normalise a count matrix
#'
#' Dispatches to one of the normalisation helpers by name.
#'
#' @param counts A numeric matrix of raw counts, features in rows.
#' @param method Normalisation method.
#' @param lengths Feature lengths in bases, required for `"rpkm"` and `"tpm"`.
#' @param norm_factors Optional normalisation factors, one per sample.
#' @return A numeric matrix of the same shape as `counts`.
#' @family normalisation
#' @export
normalise_counts <- function(counts,
                             method = c("cpm", "log_cpm", "rpkm", "tpm", "ratio"),
                             lengths = NULL,
                             norm_factors = NULL) {
  method <- match.arg(method)
  if (method %in% c("rpkm", "tpm") && is.null(lengths)) {
    stop(sprintf("method '%s' requires `lengths`", method), call. = FALSE)
  }
  switch(method,
    cpm = cpm(counts, norm_factors = norm_factors),
    log_cpm = log_cpm(counts, norm_factors = norm_factors),
    rpkm = rpkm(counts, lengths, norm_factors = norm_factors),
    tpm = tpm(counts, lengths),
    ratio = sweep(counts, 2L, size_factors_ratio(counts), "/")
  )
}

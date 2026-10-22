#' Compute per-sample QC metrics
#'
#' Summarises library size, detected features and the fraction of counts
#' from mitochondrial and spike-in features for every sample.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @param mito_pattern Regular expression matching mitochondrial feature names.
#' @param spike_pattern Regular expression matching spike-in feature names.
#' @return A data frame with one row per sample.
#' @export
qc_sample_metrics <- function(x,
                              assay_name = "counts",
                              mito_pattern = "^MT-",
                              spike_pattern = "^ERCC-") {
  counts <- assay(x, assay_name)
  features <- rownames(counts)
  lib_size <- colSums(counts)
  is_mito <- grepl(mito_pattern, features, ignore.case = TRUE)
  is_spike <- grepl(spike_pattern, features)
  mito <- if (any(is_mito)) colSums(counts[is_mito, , drop = FALSE]) else 0
  spike <- if (any(is_spike)) colSums(counts[is_spike, , drop = FALSE]) else 0
  data.frame(
    sample = colnames(counts),
    lib_size = lib_size,
    n_detected = colSums(counts > 0),
    pct_mito = ifelse(lib_size > 0, 100 * mito / lib_size, NA_real_),
    pct_spike = ifelse(lib_size > 0, 100 * spike / lib_size, NA_real_),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

#' Compute per-feature QC metrics
#'
#' Summarises total counts, mean expression and detection rate for every
#' feature.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @return A data frame with one row per feature.
#' @export
qc_feature_metrics <- function(x, assay_name = "counts") {
  counts <- assay(x, assay_name)
  n_samples <- ncol(counts)
  data.frame(
    feature = rownames(counts),
    total = rowSums(counts),
    mean = rowMeans(counts),
    n_detected = rowSums(counts > 0),
    pct_detected = 100 * rowSums(counts > 0) / n_samples,
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

#' Flag features with too few counts
#'
#' Keeps features reaching `min_count` in at least `min_samples` samples.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @param min_count Minimum count a sample must reach.
#' @param min_samples Minimum number of samples reaching `min_count`.
#' @return A logical vector with one element per feature, `TRUE` to keep.
#' @family qc filters
#' @export
keep_by_count <- function(x, assay_name = "counts", min_count = 10, min_samples = 2) {
  counts <- assay(x, assay_name)
  if (min_samples > ncol(counts)) {
    warning("`min_samples` exceeds the number of samples; no feature passes")
  }
  keep <- rowSums(counts >= min_count) >= min_samples
  names(keep) <- rownames(counts)
  keep
}

#' Flag features detected in too few samples
#'
#' Keeps features with a non-zero count in at least `min_fraction` of samples.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @param min_fraction Minimum fraction of samples with a non-zero count.
#' @return A logical vector with one element per feature, `TRUE` to keep.
#' @family qc filters
#' @export
keep_by_detection <- function(x, assay_name = "counts", min_fraction = 0.25) {
  counts <- assay(x, assay_name)
  if (min_fraction < 0 || min_fraction > 1) {
    stop("`min_fraction` must lie in [0, 1]", call. = FALSE)
  }
  keep <- rowMeans(counts > 0) >= min_fraction
  names(keep) <- rownames(counts)
  keep
}

#' Flag samples with outlying library size
#'
#' Marks samples whose log library size lies more than `nmads` median
#' absolute deviations below the median, optionally within batches.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @param nmads Number of MADs below the median that triggers removal.
#' @param batch Optional `colInfo` column defining batches.
#' @return A logical vector with one element per sample, `TRUE` to keep.
#' @family qc filters
#' @export
keep_by_libsize <- function(x, assay_name = "counts", nmads = 3, batch = NULL) {
  metrics <- qc_sample_metrics(x, assay_name)
  log_lib <- log10(metrics$lib_size + 1)
  groups <- if (is.null(batch)) {
    rep("all", length(log_lib))
  } else {
    as.character(colInfo(x, batch)[[batch]])
  }
  keep <- logical(length(log_lib))
  for (g in unique(groups)) {
    idx <- which(groups == g)
    centre <- stats::median(log_lib[idx])
    spread <- stats::mad(log_lib[idx])
    if (spread == 0 || length(idx) < 3L) {
      keep[idx] <- TRUE
    } else {
      keep[idx] <- log_lib[idx] >= centre - nmads * spread
    }
  }
  names(keep) <- metrics$sample
  keep
}

#' Flag samples with a high mitochondrial fraction
#'
#' Marks samples whose mitochondrial percentage exceeds `max_pct`.
#'
#' @param x An `AssayShelf` object.
#' @param assay_name Name of the count assay to use.
#' @param max_pct Maximum allowed percentage of mitochondrial counts.
#' @param mito_pattern Regular expression matching mitochondrial feature names.
#' @return A logical vector with one element per sample, `TRUE` to keep.
#' @family qc filters
#' @export
keep_by_mito <- function(x, assay_name = "counts", max_pct = 10, mito_pattern = "^MT-") {
  metrics <- qc_sample_metrics(x, assay_name, mito_pattern = mito_pattern)
  keep <- !is.na(metrics$pct_mito) & metrics$pct_mito <= max_pct
  names(keep) <- metrics$sample
  keep
}

#' Apply a QC filter to an AssayShelf
#'
#' Runs one of the `keep_by_*()` filters and subsets features or samples
#' accordingly, recording the outcome in the object metadata.
#'
#' @param x An `AssayShelf` object.
#' @param filter Which filter to apply.
#' @param ... Passed to the chosen `keep_by_*()` function.
#' @param verbose Should a one-line summary be printed?
#' @return The filtered `AssayShelf` object.
#' @family qc filters
#' @export
apply_qc_filter <- function(x,
                            filter = c("count", "detection", "libsize", "mito"),
                            ...,
                            verbose = TRUE) {
  filter <- match.arg(filter)
  keep <- switch(filter,
    count = keep_by_count(x, ...),
    detection = keep_by_detection(x, ...),
    libsize = keep_by_libsize(x, ...),
    mito = keep_by_mito(x, ...)
  )
  margin <- if (filter %in% c("count", "detection")) "features" else "samples"
  out <- if (margin == "features") x[keep, ] else x[, keep]
  log_entry <- data.frame(
    filter = filter,
    margin = margin,
    n_before = length(keep),
    n_removed = sum(!keep),
    stringsAsFactors = FALSE
  )
  out@meta$qc_log <- rbind(out@meta$qc_log, log_entry)
  if (verbose) {
    message(sprintf(
      "%s filter removed %d of %d %s",
      filter, sum(!keep), length(keep), margin
    ))
  }
  out
}

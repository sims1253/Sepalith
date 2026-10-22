# skipped: apply_qc_filter() needs the AssayShelf class and `[` method
# from classes.R / accessors.R
# Stand-in accessor: a plain named list of matrices plays the container.
assay <- function(x, i) x[[i]]
counts <- matrix(
  c(50, 60, 55, 0,
    0, 0, 1, 0,
    12, 15, 9, 11,
    30, 2, 25, 400,
    5, NA, 0, 6),
  nrow = 5, byrow = TRUE,
  dimnames = list(c("ACTB", "LOWEXP", "GAPDH", "MT-CO1", "TP53"), paste0("s", 1:4))
)
x <- list(counts = counts)

sm <- qc_sample_metrics(x)
stopifnot(
  identical(sm$sample, paste0("s", 1:4)),
  isTRUE(all.equal(sm$pct_mito[4], 100 * 400 / 417)),
  all(sm$pct_spike[-2] == 0),
  is.na(sm$pct_spike[2])
)

fm <- qc_feature_metrics(x)
stopifnot(
  identical(fm$feature, rownames(counts)),
  fm$n_detected[3] == 4,
  fm$pct_detected[2] == 25
)

kc <- keep_by_count(x, min_count = 10, min_samples = 2)
stopifnot(
  identical(unname(kc), c(TRUE, FALSE, TRUE, TRUE, FALSE)),
  !anyNA(kc)
)

kd <- keep_by_detection(x, min_fraction = 0.5)
stopifnot(
  identical(unname(kd), c(TRUE, FALSE, TRUE, TRUE, TRUE)),
  inherits(try(keep_by_detection(x, min_fraction = 2), silent = TRUE), "try-error")
)

stopifnot(
  identical(unname(keep_by_mito(x, max_pct = 50)), c(TRUE, FALSE, TRUE, FALSE)),
  all(keep_by_libsize(list(counts = counts[, 1:2])))
)
lib <- cbind(matrix(100, 2, 6), c(1, 0))
dimnames(lib) <- list(c("a", "b"), paste0("s", 1:7))
lib[, 1:6] <- lib[, 1:6] + 1:12
stopifnot(identical(unname(keep_by_libsize(list(counts = lib))), c(rep(TRUE, 6), FALSE)))

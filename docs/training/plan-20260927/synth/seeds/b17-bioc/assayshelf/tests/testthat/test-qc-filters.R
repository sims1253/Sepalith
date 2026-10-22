make_shelf <- function() {
  counts <- matrix(
    c(
      50, 60, 55, 0,
      0, 0, 1, 0,
      12, 15, 9, 11,
      30, 2, 25, 400,
      5, 8, 0, 6
    ),
    nrow = 5,
    byrow = TRUE,
    dimnames = list(
      c("ACTB", "LOWEXP", "GAPDH", "MT-CO1", "TP53"),
      paste0("s", 1:4)
    )
  )
  AssayShelf(
    counts,
    colInfo = data.frame(batch = c("a", "a", "b", "b"))
  )
}

test_that("qc_sample_metrics reports one row per sample", {
  shelf <- make_shelf()
  m <- qc_sample_metrics(shelf)
  expect_equal(nrow(m), 4L)
  expect_equal(m$sample, paste0("s", 1:4))
  expect_equal(m$lib_size, c(97, 85, 90, 417))
  expect_equal(m$pct_mito[4], 100 * 400 / 417)
})

test_that("qc_feature_metrics reports one row per feature", {
  shelf <- make_shelf()
  m <- qc_feature_metrics(shelf)
  expect_equal(nrow(m), 5L)
  expect_equal(m$n_detected, c(3, 1, 4, 4, 3))
  expect_equal(m$pct_detected[2], 25)
})

test_that("keep_by_count drops features below the threshold", {
  shelf <- make_shelf()
  keep <- keep_by_count(shelf, min_count = 10, min_samples = 2)
  expect_named(keep, rownames(assay(shelf)))
  expect_false(keep[["LOWEXP"]])
  expect_false(keep[["TP53"]])
  expect_true(keep[["ACTB"]])
})

test_that("keep_by_detection drops rarely detected features", {
  shelf <- make_shelf()
  keep <- keep_by_detection(shelf, min_fraction = 0.5)
  expect_named(keep, rownames(assay(shelf)))
  expect_false(keep[["LOWEXP"]])
  expect_true(keep[["TP53"]])
  expect_error(keep_by_detection(shelf, min_fraction = 2), "must lie in")
})

test_that("keep_by_mito drops samples dominated by mitochondrial reads", {
  shelf <- make_shelf()
  keep <- keep_by_mito(shelf, max_pct = 50)
  expect_named(keep, paste0("s", 1:4))
  expect_equal(unname(keep), c(TRUE, TRUE, TRUE, FALSE))
})

test_that("keep_by_libsize keeps everything in tiny batches", {
  shelf <- make_shelf()
  keep <- keep_by_libsize(shelf, batch = "batch")
  expect_true(all(keep))
  expect_named(keep, paste0("s", 1:4))
})

test_that("apply_qc_filter subsets features and logs the step", {
  shelf <- make_shelf()
  out <- apply_qc_filter(shelf, "count", min_count = 10, verbose = FALSE)
  expect_equal(nrow(assay(out)), 3L)
  expect_equal(out@meta$qc_log$filter, "count")
  expect_equal(out@meta$qc_log$n_removed, 2L)
})

test_that("apply_qc_filter subsets samples and logs the step", {
  shelf <- make_shelf()
  out <- apply_qc_filter(shelf, "mito", max_pct = 50, verbose = FALSE)
  expect_equal(ncol(assay(out)), 3L)
  expect_equal(out@meta$qc_log$filter, "mito")
  expect_equal(out@meta$qc_log$n_removed, 1L)
  expect_message(apply_qc_filter(shelf, "mito", max_pct = 50), "removed 1 of 4 samples")
})

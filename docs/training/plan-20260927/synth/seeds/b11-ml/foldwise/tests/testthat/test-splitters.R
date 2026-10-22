test_that("kfold_splits covers every row exactly once in assessment", {
  splits <- kfold_splits(23, k = 5, seed = 42)
  held_out <- unlist(lapply(splits, `[[`, "assessment"))
  expect_s3_class(splits, "fw_splits")
  expect_length(splits, 5)
  expect_setequal(held_out, 1:23)
  expect_false(anyDuplicated(held_out) > 0)
})

test_that("kfold_splits keeps analysis and assessment disjoint", {
  splits <- kfold_splits(30, k = 3, seed = 7)
  for (fold in splits) {
    expect_length(intersect(fold$analysis, fold$assessment), 0)
    expect_equal(length(fold$analysis) + length(fold$assessment), 30)
  }
})

test_that("kfold_splits rejects more folds than rows", {
  expect_error(kfold_splits(3, k = 5), "Cannot make 5 folds from 3 rows")
  expect_error(kfold_splits(10, k = 1), "at least 2")
  expect_error(kfold_splits(NA, k = 2), "single positive number")
})

test_that("stratified_splits roughly preserves class balance", {
  strata <- rep(c("yes", "no"), times = c(20, 80))
  splits <- stratified_splits(strata, k = 4, seed = 1)
  shares <- vapply(splits, function(f) mean(strata[f$assessment] == "yes"), numeric(1))
  expect_true(all(abs(shares - 0.2) < 0.05))
  expect_setequal(unlist(lapply(splits, `[[`, "assessment")), seq_along(strata))
})

test_that("grouped_splits never splits a group across folds", {
  groups <- rep(letters[1:9], times = c(5, 3, 8, 2, 6, 4, 7, 1, 4))
  splits <- grouped_splits(groups, k = 3, seed = 11)
  for (fold in splits) {
    expect_length(intersect(groups[fold$analysis], groups[fold$assessment]), 0)
  }
  expect_length(splits, 3)
})

test_that("logo_splits makes one fold per group", {
  groups <- c("x", "y", "x", "z", "y", "z", "z")
  splits <- logo_splits(groups)
  expect_length(splits, 3)
  expect_equal(splits[[1]]$assessment, c(1L, 3L))
  expect_error(logo_splits(rep("only", 4)), "at least two groups")
})

test_that("rolling_splits moves forward in time", {
  splits <- rolling_splits(12, initial = 6, assess = 2, cumulative = FALSE)
  expect_length(splits, 3)
  expect_equal(splits[[1]]$analysis, 1:6)
  expect_equal(splits[[2]]$analysis, 3:8)
  expect_equal(splits[[3]]$assessment, 11:12)
  expect_error(rolling_splits(5, initial = 4, assess = 2), "must not exceed")
})

test_that("print.fw_splits reports method and sizes", {
  splits <- kfold_splits(10, k = 2, seed = 3)
  expect_output(print(splits), "<fw_splits: kfold, 2 folds, n = 10>")
  expect_output(print(splits), "Fold01")
  expect_invisible(print(splits))
})

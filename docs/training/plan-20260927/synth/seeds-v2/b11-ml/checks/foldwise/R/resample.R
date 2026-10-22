df <- data.frame(x = 1:6, y = c(2, 4, 6, 8, 10, 12))
fold <- list(id = "Fold01", analysis = c(1L, 2L, 3L, 4L), assessment = c(5L, 6L))

# fold_analysis / fold_assessment
stopifnot(identical(fold_analysis(df, fold)$x, 1:4), identical(fold_assessment(df, fold)$x, 5:6))
stopifnot(inherits(try(fold_assessment(df[1:3, ], fold), silent = TRUE), "try-error"))

# run_one_fold
fit_mean <- function(d) mean(d$y)
pred_mean <- function(m, d) rep(m, nrow(d))
res <- run_one_fold(df, fold, "y", fit_mean, pred_mean)
stopifnot(identical(res$.row, c(5L, 6L)), all(res$estimate == 5), identical(res$truth, c(10, 12)))

# cross_validate
splits <- structure(
  list(fold, list(id = "Fold02", analysis = 3:6, assessment = 1:2)),
  method = "manual", class = "fw_splits"
)
oof <- cross_validate(df, splits, "y", fit_mean, pred_mean)
stopifnot(inherits(oof, "fw_oof"), nrow(oof) == 4, identical(attr(oof, "method"), "manual"))
bad_fit <- function(d) stop("boom")
stopifnot(inherits(try(suppressWarnings(cross_validate(df, splits, "y", bad_fit)),
  silent = TRUE), "try-error"))

# score_folds / summarise_scores
mae <- function(t, e) mean(abs(t - e))
sc <- score_folds(oof, mae)
stopifnot(identical(sc$id, c("Fold01", "Fold02")), all.equal(sc$value, c(6, 6)))
sm <- summarise_scores(sc)
stopifnot(sm$mean == 6, sm$std_err == 0, sm$n == 2)
stopifnot(summarise_scores(data.frame(id = "a", value = NA))$n == 0)

# oof_vector
v <- oof_vector(oof, 6)
stopifnot(identical(v[c(1, 5)], c(9, 5)), all(is.na(v[3:4])))
dup <- data.frame(.row = c(1L, 1L), estimate = c(2, 4))
stopifnot(identical(oof_vector(dup, 2), c(3, NA)))

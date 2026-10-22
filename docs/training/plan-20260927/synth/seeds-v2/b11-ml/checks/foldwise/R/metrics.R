truth <- c(1, 2, 3, NA)
est <- c(1, 4, 3, 5)

# align_pairs
p <- align_pairs(truth, est)
stopifnot(identical(p$truth, c(1, 2, 3)), length(align_pairs(truth, est, FALSE)$truth) == 4)
stopifnot(inherits(try(align_pairs(1:2, 1), silent = TRUE), "try-error"))

# rmse_vec / mae_vec / mape_vec / rsq_trad_vec
stopifnot(all.equal(rmse_vec(truth, est), sqrt(4 / 3)), all.equal(mae_vec(truth, est), 2 / 3))
stopifnot(all.equal(mape_vec(c(0, 2), c(1, 1)), 50), is.na(mape_vec(0, 1)))
stopifnot(rsq_trad_vec(c(1, 2, 3), c(1, 2, 3)) == 1, is.na(rsq_trad_vec(c(2, 2), c(1, 3))))

# log_loss_vec / brier_vec
stopifnot(all.equal(log_loss_vec(c(1, 0), c(0.5, 0.5)), log(2)))
stopifnot(inherits(try(log_loss_vec(c(2, 0), c(0.5, 0.5)), silent = TRUE), "try-error"))
stopifnot(all.equal(brier_vec(c(TRUE, FALSE), c(0.8, 0.4)), (0.04 + 0.16) / 2))

# auc_vec
stopifnot(auc_vec(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9)) == 1)
stopifnot(auc_vec(c(0, 1), c(0.5, 0.5)) == 0.5)
stopifnot(is.na(suppressWarnings(auc_vec(c(1, 1), c(0.2, 0.3)))))

# metric_table
tab <- metric_table(c(1, 2), c(1, 2))
stopifnot(identical(tab$metric, c("rmse", "mae", "rsq")), all(tab$value[1:2] == 0))
stopifnot(inherits(try(metric_table(1, 1, list(rmse_vec)), silent = TRUE), "try-error"))

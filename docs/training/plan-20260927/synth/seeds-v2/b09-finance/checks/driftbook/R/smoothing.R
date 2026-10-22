y <- c(10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16, 18)

# ses_filter
s1 <- ses_filter(c(1, 2, 3), 1)
stopifnot(all.equal(s1$fitted, c(1, 1, 2)), s1$sse == 2, inherits(s1, "drift_smooth"))
stopifnot(all.equal(ses_filter(c(4, 4), 0.3, level0 = 2)$fitted[1], 2))

# holt_filter
h1 <- holt_filter(c(1, 2, 3, 4), 1, 1)
stopifnot(h1$fitted[1] == 2, all.equal(h1$level, 1:4), all.equal(h1$trend, c(0, 1, 1, 1)))
stopifnot(inherits(try(holt_filter(1:2, 0.5, 0.5), silent = TRUE), "try-error"))

# fit_smoother
best <- fit_smoother(y, "ses")
stopifnot(best$method == "ses", best$sse <= ses_filter(y, 0.5)$sse)
hb <- fit_smoother(y, "holt", grid = c(0.2, 0.8))
stopifnot(hb$method == "holt", hb$sse <= holt_filter(y, 0.2, 0.8)$sse)

# predict.drift_smooth / print.drift_smooth
stopifnot(all.equal(predict(h1, 3), 4 + 1:3), length(predict(s1, 2)) == 2)
out <- capture.output(res <- print(h1))
stopifnot(identical(res, h1), grepl("Holt", out[1]))

# rolling_accuracy
acc <- rolling_accuracy(y, "ses", h = 1, min_train = 10)
stopifnot(all(c("mae", "rmse", "mape") %in% names(acc)), acc[["rmse"]] >= acc[["mae"]])
stopifnot(inherits(try(rolling_accuracy(1:5), silent = TRUE), "try-error"))

# check_weight
stopifnot(identical(check_weight(0.4, "a"), 0.4))
stopifnot(inherits(try(check_weight(2, "a"), silent = TRUE), "try-error"))

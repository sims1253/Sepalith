x <- c(1, 2, 3, 4, 5, 100, NA)

f1 <- outlier.fences(x, "iqr")
stopifnot(identical(names(f1), c("lower", "upper")), isTRUE(all.equal(unname(f1), c(-1.5, 8.5))))
f2 <- outlier.fences(c(1, 2, 3), "zscore", k = 1)
stopifnot(isTRUE(all.equal(unname(f2), c(1, 3))))
f3 <- outlier.fences(c(1, 2, 3, 4, 5), "mad", k = 1)
stopifnot(isTRUE(all.equal(unname(f3), c(3 - 1.4826, 3 + 1.4826))))
f4 <- outlier.fences(0:100, "quantile", k = 0.1)
stopifnot(isTRUE(all.equal(unname(f4), c(10, 90))))
stopifnot(isTRUE(all.equal(unname(outlier.fences(0:100, "quantile")), c(1, 99))))

stopifnot(identical(flag.outliers(x), c(FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE)))
stopifnot(!any(flag.outliers(x, "zscore")))

w <- winsorize(c(-5, 0, 5, 10), lower = 0, upper = 6)
stopifnot(identical(as.vector(w), c(0, 0, 5, 6)))
stopifnot(identical(attr(w, "clipped"), list(lower = 1L, upper = 4L)))
stopifnot(identical(as.vector(winsorize(c(-5, 10), upper = 0)), c(-5, 0)))

stopifnot(identical(as.vector(treat.outliers(x, action = "flag")), x))
stopifnot(identical(attr(treat.outliers(x), "outliers"), 6L))
stopifnot(identical(as.vector(treat.outliers(x, action = "na")), c(1, 2, 3, 4, 5, NA, NA)))
stopifnot(identical(as.vector(treat.outliers(x, action = "clip")), c(1, 2, 3, 4, 5, 8.5, NA)))
stopifnot(identical(as.vector(treat.outliers(x, action = "drop")), c(1, 2, 3, 4, 5, NA)))

df <- data.frame(a = x, b = c(10, 10, 11, 12, 10, 11, 10), s = letters[1:7])
rep <- outlier.report(df)
stopifnot(identical(rep$column, c("a", "b")), identical(rep$n.high, c(1L, 0L)))
stopifnot(identical(rep$n.missing, c(1L, 0L)), identical(rep$n.low, c(0L, 0L)))
stopifnot(identical(names(rep), c("column", "n", "n.missing", "lower", "upper", "n.low",
                                  "n.high")))

cl <- clean.numeric.columns(df)
stopifnot(identical(cl$treated, c(a = 1L, b = 0L)), is.na(cl$data$a[6]))
cl2 <- clean.numeric.columns(df, columns = "a", action = "clip")
stopifnot(identical(cl2$data$a[6], 8.5), identical(names(cl2$treated), "a"))

fl <- outlier.fences(c(1, 10, 100, 1000, 10000), "log.iqr", k = 0)
stopifnot(identical(names(fl), c("lower", "upper")), isTRUE(all.equal(unname(fl), c(10, 1000))))
stopifnot(identical(flag.outliers(c(1, 10, 100, 1000, 1e9), "log.iqr"),
                    c(FALSE, FALSE, FALSE, FALSE, TRUE)))
stopifnot(inherits(try(winsorize(1:3, lower = 3, upper = 1), silent = TRUE), "try-error"))

vit <- data.frame(heart.rate = c(70, 400, NA), systolic = c(120, 30, 110),
                  diastolic = c(80, 70, 115), temperature = c(37, 98.6, 50),
                  weight = c(70, 0, 500))
cv <- clean.vitals(vit)
stopifnot(identical(names(cv), c("data", "removed")))
stopifnot(identical(cv$removed, c(heart.rate = 1L, systolic = 1L, diastolic = 1L,
                                  temperature = 1L, weight = 2L)))
stopifnot(identical(cv$data$heart.rate, c(70, NA, NA)))
stopifnot(identical(cv$data$systolic, c(120, NA, 110)))
stopifnot(identical(cv$data$diastolic, c(80, 70, NA)))
stopifnot(isTRUE(all.equal(cv$data$temperature, c(37, 37, NA))))
stopifnot(identical(cv$data$weight, c(70, NA, NA)))
stopifnot(inherits(try(clean.numeric.columns(df, columns = "zz"), silent = TRUE), "try-error"))

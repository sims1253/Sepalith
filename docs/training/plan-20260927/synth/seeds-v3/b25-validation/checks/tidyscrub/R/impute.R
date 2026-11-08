fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

stopifnot(identical(fill.forward(c(NA, 1, NA, NA, 4, NA)), c(NA, 1, 1, 1, 4, 4)))
stopifnot(identical(fill.forward(c(1, NA, NA, NA, 5), max.gap = 2), c(1, 1, 1, NA, 5)))
stopifnot(identical(fill.forward(c(NA, NA)), c(NA, NA)))
stopifnot(identical(fill.backward(c(NA, 1, NA, NA, 4, NA)), c(1, 1, 4, 4, 4, NA)))
stopifnot(identical(fill.backward(c("a", NA, "b")), c("a", "b", "b")))

stopifnot(identical(most.frequent(c("b", "a", "a", NA)), "a"))
stopifnot(identical(most.frequent(c(2, 1)), 2))
stopifnot(is.na(most.frequent(c(NA, NA))))

x <- c(1, NA, 3, NA)
m <- impute.column(x, "mean")
stopifnot(identical(as.vector(m), c(1, 2, 3, 2)), identical(attr(m, "imputed"), c(2L, 4L)))
stopifnot(identical(as.vector(impute.column(c(1, NA, 3, 10), "median")), c(1, 3, 3, 10)))
stopifnot(identical(as.vector(impute.column(c("a", NA, "a", "b"), "mode")), c("a", "a", "a", "b")))
stopifnot(identical(as.vector(impute.column(x, "constant", value = 0)), c(1, 0, 3, 0)))
stopifnot(identical(as.vector(impute.column(x, "locf")), c(1, 1, 3, 3)))
n <- impute.column(x, "nocb")
stopifnot(identical(as.vector(n), c(1, 3, 3, NA)), identical(attr(n, "imputed"), 2L))
stopifnot(fails(impute.column("a", "mean")), fails(impute.column("a", "median")))
stopifnot(fails(impute.column(x, "constant")))

df <- data.frame(g = c("a", "a", "b", "b"), v = c(1, NA, NA, 7), w = c("x", NA, "y", "y"),
                 stringsAsFactors = FALSE)
res <- impute.frame(df, c(v = "mean", w = "mode"), flag = TRUE)
stopifnot(identical(names(res), c("data", "n.imputed")))
stopifnot(identical(res$n.imputed, c(v = 2L, w = 1L)))
stopifnot(identical(res$data$v, c(1, 4, 4, 7)), identical(res$data$w, c("x", "y", "y", "y")))
stopifnot(identical(res$data$v.imputed, c(FALSE, TRUE, TRUE, FALSE)))
stopifnot(is.null(impute.frame(df, c(v = "locf"))$data$v.imputed))
stopifnot(fails(impute.frame(df, c(z = "mean"))))

grp <- impute.by.group(df, "v", "g", method = "mean")
stopifnot(identical(grp$v, c(1, 1, 7, 7)))
df2 <- df
df2$v[1] <- NA
stopifnot(identical(impute.by.group(df2, "v", "g")$v, c(NA, NA, 7, 7)))

li <- impute.column(c(1, NA, 3, NA, NA, 9, NA), "linear")
stopifnot(identical(as.vector(li), c(1, 2, 3, 5, 7, 9, NA)))
stopifnot(identical(attr(li, "imputed"), c(2L, 4L, 5L)))
stopifnot(fails(impute.column("a", "linear")))
stopifnot(fails(impute.by.group(df, "v", "grp")))

sdf <- data.frame(sym = c(1, 2, 3, NA, 2, 3, 1, 2),
                  skw = c(1, 1, 1, 1, NA, 1, 1, 60),
                  full = 1:8,
                  s = c("a", NA, "a", "b", "a", "a", "b", "a"),
                  l = c(TRUE, NA, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE),
                  stringsAsFactors = FALSE)
sdf$f <- factor(sdf$s)
sdf$d <- as.Date("2024-01-01") + c(0:6, NA)
sug <- suggest.imputation(sdf)
stopifnot(identical(sug, c(sym = "mean", skw = "median", s = "mode", l = "mode", f = "mode",
                           d = "locf")))
stopifnot(identical(suggest.imputation(sdf, skew.cutoff = 5)[["skw"]], "mean"))
stopifnot(identical(suggest.imputation(data.frame(e = NA, k = 1)), character()))
stopifnot(identical(suggest.imputation(data.frame(z = I(list(1, NA)))), c(z = "constant")))

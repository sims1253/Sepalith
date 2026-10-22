df <- data.frame(a = c(1, 2, 3, 4), b = c(10, 10, 10, 10), s = c("x", "y", "x", "y"))

# resolveNumericCols
stopifnot(identical(resolveNumericCols(df), c("a", "b")))
stopifnot(inherits(try(resolveNumericCols(df, "zz"), silent = TRUE), "try-error"))
stopifnot(inherits(try(resolveNumericCols(df, "s"), silent = TRUE), "try-error"))

# fitStandardScaler / fitMinMaxScaler / fitRobustScaler
sc <- fitStandardScaler(df)
stopifnot(inherits(sc, "fp_scaler"), sc$center[["a"]] == 2.5, sc$scale[["b"]] == 1)
mm <- fitMinMaxScaler(df, "a")
stopifnot(mm$center[["a"]] == 1, mm$scale[["a"]] == 3, mm$type == "minmax")
rb <- fitRobustScaler(df, "a")
stopifnot(rb$center[["a"]] == 2.5, rb$scale[["a"]] == IQR(df$a))

# applyScaler / invertScaler
out <- applyScaler(mm, df)
stopifnot(all.equal(out$a, c(0, 1 / 3, 2 / 3, 1)), identical(out$s, df$s))
stopifnot(all.equal(invertScaler(mm, out)$a, df$a))
stopifnot(inherits(try(applyScaler(mm, df["b"]), silent = TRUE), "try-error"))

# print.fp_scaler
txt <- capture.output(res <- print(sc))
stopifnot(grepl("<fp_scaler: standard on 2 columns>", txt[1], fixed = TRUE), identical(res, sc))

# writeScaler / readScaler
path <- tempfile(fileext = ".csv")
stopifnot(identical(writeScaler(rb, path), path))
back <- readScaler(path)
stopifnot(identical(back$type, "robust"), all.equal(back$center, rb$center))
bad <- tempfile(fileext = ".csv")
write.csv(data.frame(x = 1), bad, row.names = FALSE)
stopifnot(inherits(try(readScaler(bad), silent = TRUE), "try-error"))

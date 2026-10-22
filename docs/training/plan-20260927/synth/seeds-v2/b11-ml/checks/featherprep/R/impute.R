df <- data.frame(
  n = c(1, NA, 3, 10),
  f = factor(c("a", "b", NA, "a")),
  l = c(TRUE, NA, TRUE, FALSE),
  g = c("u", "u", "v", NA),
  empty = c(NA_real_, NA, NA, NA),
  stringsAsFactors = FALSE
)

# modeValue
stopifnot(identical(modeValue(c("b", "a", "a", "b")), "b"))
stopifnot(is.na(modeValue(c(NA, NA))), identical(modeValue(c(2, NA, 2)), 2))

# fitImputer / applyImputer
imp <- fitImputer(df)
stopifnot(imp$fills$n == 3, identical(imp$fills$f, "a"), identical(imp$fills$l, TRUE))
out <- applyImputer(imp, df)
stopifnot(out$n[2] == 3, identical(as.character(out$f[3]), "a"))
stopifnot(is.logical(out$l), isTRUE(out$l[2]), all(is.na(out$empty)))
stopifnot(fitImputer(df, "n", method = "mean")$fills$n == mean(c(1, 3, 10)))

# addMissingIndicators
ind <- addMissingIndicators(df, c("n", "g"))
stopifnot(identical(ind$n_missing, c(0L, 1L, 0L, 0L)), identical(ind$g_missing, c(0L, 0L, 0L, 1L)))
stopifnot(!("x_missing" %in% names(addMissingIndicators(data.frame(x = 1:3)))))

# imputeByGroup
grp <- data.frame(y = c(1, NA, 5, NA, NA), k = c("a", "a", "b", "c", NA))
filled <- imputeByGroup(grp, "y", "k")
stopifnot(identical(filled$y, c(1, 1, 5, 3, 3)))

# missingProfile / dropSparseColumns
prof <- missingProfile(df)
stopifnot(identical(prof$column[1], "empty"), prof$share[1] == 1, nrow(prof) == 5)
stopifnot(nrow(missingProfile(data.frame())) == 0)
stopifnot(identical(names(dropSparseColumns(df, 0.5)), c("n", "f", "l", "g")))
stopifnot(inherits(try(dropSparseColumns(df, 2), silent = TRUE), "try-error"))

# print.fp_imputer
txt <- capture.output(print(imp))
stopifnot(grepl("method = median", txt[1], fixed = TRUE), length(txt) == 6)

df <- data.frame(col = c("a", "b", "a", "c", NA), stringsAsFactors = FALSE)

# fitOneHot
enc <- fitOneHot(df, "col")
stopifnot(identical(enc$levels$col, c("a", "b", "c")))
pooled <- fitOneHot(df, "col", minCount = 2L)
stopifnot(identical(pooled$levels$col, c("a", "other")))

# dummyNames / mapToLevels
stopifnot(identical(dummyNames("col", c("a", "b c")), c("col_a", "col_b.c")))
stopifnot(identical(mapToLevels(c("a", "z", NA), c("a", "other")), c("a", "other", NA)))
stopifnot(identical(mapToLevels(c("a", "z"), c("a", "b")), c("a", NA)))

# applyOneHot
out <- applyOneHot(enc, df)
stopifnot(identical(names(out), c("col_a", "col_b", "col_c")))
stopifnot(identical(out$col_a, c(1L, 0L, 1L, 0L, 0L)))
stopifnot(is.na(applyOneHot(enc, df, keepNA = TRUE)$col_a[5]))
stopifnot(nrow(applyOneHot(enc, df[0, , drop = FALSE])) == 0)
one <- applyOneHot(pooled, data.frame(col = "b", stringsAsFactors = FALSE))
stopifnot(identical(unlist(one, use.names = FALSE), c(0L, 1L)))

# fitLabelEncoder / applyLabelEncoder
lab <- fitLabelEncoder(data.frame(k = c("b", "a", "b")), "k", order = "frequency")
stopifnot(identical(lab$levels$k, c("b", "a")))
coded <- applyLabelEncoder(lab, data.frame(k = c("a", "z", NA)))
stopifnot(identical(coded$k, c(2L, 0L, NA)))

# print methods
stopifnot(grepl("1 columns -> 3 dummies", capture.output(print(enc))[1], fixed = TRUE))
stopifnot(grepl("order = frequency", capture.output(print(lab))[1], fixed = TRUE))

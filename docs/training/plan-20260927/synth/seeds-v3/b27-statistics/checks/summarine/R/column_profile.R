stopifnot(column_type(1:3) == "numeric", column_type(c(TRUE, NA)) == "logical")
stopifnot(column_type(factor("a")) == "factor", column_type("a") == "character")
stopifnot(column_type(list(1)) == "other")

pn <- profile_numeric(c(1, 2, 3, NA), "v")
stopifnot(pn$column == "v", pn$n == 4L, pn$n_missing == 1L, pn$center == 2, pn$spread == 1)

pl <- profile_logical(c(TRUE, FALSE, TRUE, TRUE))
stopifnot(pl$type == "logical", pl$center == 0.75, all.equal(pl$spread, sqrt(0.75 * 0.25)))

pf <- profile_factor(factor(c("a", "a", "b", "b", NA)))
stopifnot(pf$center == 0.5, all.equal(pf$spread, 1), pf$n_distinct == 2L)
stopifnot(profile_factor(factor(c("a", "a")))$spread == 0)

pc <- profile_character(c("ab", "abcd", NA))
stopifnot(pc$center == 3, all.equal(pc$spread, sqrt(2)), pc$n_missing == 1L)

df <- data.frame(a = c(1, NA, 3), b = c(TRUE, FALSE, NA), f = factor(c("x", "y", "x")),
  s = c("q", "rr", "q"), stringsAsFactors = FALSE)
pr <- profile_columns(df)
stopifnot(identical(pr$type, c("numeric", "logical", "factor", "character")), nrow(pr) == 4L)
df$l <- I(list(1, 2, 3))
stopifnot(inherits(try(profile_columns(df, "l"), silent = TRUE), "try-error"))

rg <- column_ranges(data.frame(u = c(3, -Inf, 1, 5, 1), w = c(NA, 2, 2, 9, 0)))
stopifnot(identical(rg$lower, c(1, 0)), identical(rg$lower_row, c(3L, 5L)))
stopifnot(identical(rg$upper, c(5, 9)), identical(rg$upper_row, c(4L, 4L)))
stopifnot(identical(rg$lower_ties, c(2L, 1L)), identical(rg$upper_ties, c(1L, 1L)))
stopifnot(identical(rg$lower_gap, c(2, 2)), identical(rg$upper_gap, c(2, 7)))
stopifnot(column_ranges(data.frame(u = c(-Inf, 2)), finite = FALSE)$lower == -Inf)
stopifnot(is.na(column_ranges(data.frame(u = c(NA_real_, NA)))$lower))
stopifnot(is.na(column_ranges(data.frame(u = c(4, 4)))$upper_gap))

b <- data.frame(a = c(1, 2, 3), f = c("x", "yy", NA), stringsAsFactors = FALSE)
a <- data.frame(a = c(2, 4, 6, NA), f = c("x", "yy", "zzz", "w"), stringsAsFactors = FALSE)
cp <- compare_profiles(b, a)
stopifnot(identical(cp$column, c("a", "f")), identical(cp$n_after, c(4L, 4L)))
stopifnot(identical(cp$missing_before, c(0L, 1L)), cp$center_change[1] == 2)
stopifnot(cp$spread_ratio[1] == 2, cp$center_before[2] == 1.5)
stopifnot(inherits(try(compare_profiles(b, a, "zz"), silent = TRUE), "try-error"))
a$a <- as.character(a$a)
w <- tryCatch(compare_profiles(b, a), warning = function(w) "warned")
stopifnot(identical(w, "warned"))

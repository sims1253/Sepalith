stopifnot(mode_value(c(3, 1, 3, 2, NA)) == 3, identical(mode_value(c("b", "a", "b", "a")), "a"))
stopifnot(identical(mode_value(c(2, 1, 2, 1), ties = "all"), c(1, 2)))
stopifnot(length(mode_value(c(NA, NA))) == 0L)

ft <- freq_table(c("b", "a", "b", "c", "b", "a", NA))
stopifnot(identical(ft$value, c("b", "a", "c")), identical(ft$cum_n, c(3L, 5L, 6L)))
stopifnot(all.equal(ft$cum_prop[3], 1), all.equal(ft$prop[1], 0.5))
stopifnot(identical(freq_table(c(2, 1, 1), sort = FALSE)$value, c("1", "2")))

xx <- c("a", "a", "b", "b", "b")
yy <- c("u", "v", "u", "u", "v")
stopifnot(prop_table_by(xx, yy)["b", "u"] == 2)
stopifnot(all.equal(prop_table_by(xx, yy, "total")["a", "u"], 0.2))
stopifnot(all.equal(rowSums(prop_table_by(xx, yy, "row")), c(a = 1, b = 1)))
stopifnot(all.equal(unname(colSums(prop_table_by(xx, yy, "column"))), c(1, 1)))
stopifnot(prop_table_by(xx, yy, "row", digits = 2)["b", "u"] == 0.67)
stopifnot(inherits(try(prop_table_by(1:2, 1), silent = TRUE), "try-error"))

cs <- crosstab_stats(rep(c("a", "b"), each = 10), rep(c("u", "v", "u", "u"), 5))
ref <- suppressWarnings(stats::chisq.test(table(rep(c("a", "b"), each = 10),
  rep(c("u", "v", "u", "u"), 5)), correct = FALSE))
stopifnot(all.equal(cs$chisq, unname(ref$statistic)), cs$df == 1L)
stopifnot(all.equal(cs$p_value, ref$p.value), all.equal(cs$phi, cs$cramer_v))
stopifnot(cs$contingency >= 0, cs$contingency < 1)

rt <- risk_table(c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, NA),
  c(TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE))
stopifnot(rt$risk_exposed == 0.75, rt$risk_unexposed == 0.25, rt$risk_ratio == 3)
stopifnot(rt$risk_diff == 0.5, rt$odds_ratio == 9)
stopifnot(all.equal(log(rt$rr_upper) - log(3), 1.959964 * sqrt(1 / 3 - 1 / 4 + 1 - 1 / 4),
  tolerance = 1e-6))
stopifnot(all.equal(rt$or_lower * rt$or_upper, 81), rt$rr_lower < 3)
stopifnot(risk_table(rep(c(TRUE, FALSE), 4), rep(c(TRUE, TRUE, FALSE, FALSE), 2),
  conf_level = 0.5)$rr_lower < 1)

bs <- binned_summary(c(1, 2, 3, 4, 5, 6), c(10, 12, 20, 22, 30, NA), breaks = c(0, 2, 4, 6))
stopifnot(identical(bs$n, c(2L, 2L, 1L)), all.equal(bs$y_mean, c(11, 21, 30)))
stopifnot(all.equal(bs$x_mean, c(1.5, 3.5, 5)), is.na(bs$y_sd[3]), all.equal(bs$y_se[1], 1))
stopifnot(identical(bs$y_min, c(10, 20, 30)), identical(bs$y_max, c(12, 22, 30)))
stopifnot(identical(bs$y_median, c(11, 21, 30)))
stopifnot(inherits(try(binned_summary(1:3, 1:2), silent = TRUE), "try-error"))

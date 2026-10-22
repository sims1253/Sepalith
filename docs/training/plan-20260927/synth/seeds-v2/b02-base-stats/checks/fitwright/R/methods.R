stopifnot(identical(format_p(c(0.5, 0.0001)), c("0.5", "< 0.001")))
stopifnot(identical(format_p(0.012345, digits = 2), "0.012"))

bt <- structure(list(t0 = c(m = 1), t = matrix(c(0.9, 1.1, NA), ncol = 1,
  dimnames = list(NULL, "m")), R = 3L, call = quote(boot_stat(x))), class = "fw_boot")
# boot_ci_percentile() lives in bootstrap.R; a fixed stub is enough here.
boot_ci_percentile <- function(boot, level) {
  cbind(lower = 0.9, upper = 1.1)
}
out <- utils::capture.output(res <- print(bt))
stopifnot(identical(res, bt), out[1] == "<fw_boot> 3 resamples", out[2] == "Call: boot_stat(x)")

s <- summary(bt, level = 0.8)
stopifnot(inherits(s, "summary.fw_boot"), s$failed == 1L, s$level == 0.8)
stopifnot(all.equal(s$table$bias, 0), s$table$lower == 0.9)
out <- utils::capture.output(print(s))
stopifnot(grepl("80% percentile CI", out[1]), grepl("1 resample", out[2]))

pm <- structure(list(statistic = 1.234567, null_dist = seq(-2, 2, by = 0.5), p_value = 0.0004,
  alternative = "two.sided", method = "Demo test", n_perm = 9L), class = "fw_perm")
out <- utils::capture.output(res <- print(pm))
stopifnot(identical(res, pm), out[1] == "Demo test", grepl("< 0.001", out[3]))

sp <- summary(pm, probs = c(0.5, 1))
stopifnot(inherits(sp, "summary.fw_perm"), all.equal(unname(sp$null_quantiles), c(0, 2)))
stopifnot(all.equal(sp$mc_error, sqrt(0.0004 * 0.9996 / 9)))
out <- utils::capture.output(res <- print(sp))
stopifnot(identical(res, sp), grepl("MC s.e.", out[3]), out[4] == "Null distribution quantiles:")

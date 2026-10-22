# kish_deff / effective_n / weight_cv
stopifnot(all.equal(kish_deff(rep(3, 5)), 1), all.equal(kish_deff(c(1, 3)), 1.25))
stopifnot(all.equal(effective_n(c(1, 1, NA, 1)), 3), all.equal(effective_n(c(1, 3)), 1.6))
stopifnot(weight_cv(rep(2, 4)) == 0, weight_cv(c(1, 3)) > 0)

# trim_weights_cap
w <- c(rep(1, 9), 30)
capped <- trim_weights_cap(w, cap = 5)
stopifnot(all.equal(sum(capped), sum(w)), max(capped) <= 5 * stats::median(w) + 1e-8)
capped_na <- trim_weights_cap(c(w, NA), cap = 5)
stopifnot(is.na(capped_na[11]), all.equal(capped_na[1:10], capped))

# trim_weights_quantile
wq <- trim_weights_quantile(w, lower = 0, upper = 0.9)
stopifnot(all.equal(sum(wq), sum(w)), max(wq) < 30)
stopifnot(is.na(trim_weights_quantile(c(w, NA))[11]))
stopifnot(inherits(try(trim_weights_quantile(w, 0.5, 0.5), silent = TRUE), "try-error"))

# weight_report / format / print
rep_obj <- weight_report(c(1, 2, NA), label = "demo")
stopifnot(inherits(rep_obj, "weight_report"), rep_obj$n == 2, rep_obj$n_missing == 1)
txt <- format(rep_obj)
stopifnot(length(txt) == 8, grepl("demo", txt[1]))
out <- utils::capture.output(res <- print(rep_obj))
stopifnot(identical(res, rep_obj), length(out) == 8)

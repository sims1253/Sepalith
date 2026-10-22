# skipped: needs lme4, dplyr, tibble (fit_trial_lmm, fit_trial_glmm,
# variance_components, icc_from_fit, fixed_effects_table,
# random_effects_table, fit_diagnostics)

# trial_formula
f <- trial_formula("hamd", "arm", "week", "pid", covariates = "hamd_bl")
stopifnot(inherits(f, "formula"))
stopifnot(identical(deparse(f), "hamd ~ arm * week + hamd_bl + (1 | pid)"))
g <- trial_formula("y", "trt", "visit", "id", random_slope = TRUE)
stopifnot(identical(deparse(g), "y ~ trt * visit + (1 + visit | id)"))
stopifnot(identical(all.vars(g)[1], "y"))

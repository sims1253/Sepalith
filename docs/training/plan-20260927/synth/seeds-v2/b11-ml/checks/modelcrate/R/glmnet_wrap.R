# skipped: needs glmnet (fit_glmnet_model, predict.mc_glmnet, glmnet_coef_table,
# glmnet_path, print.mc_glmnet)
train <- data.frame(x = c(1, 2, 3), g = factor(c("a", "b", "c")))

# glmnet_matrix
m <- glmnet_matrix(train, c("x", "g"))
stopifnot(identical(colnames(m), c("x", "gb", "gc")), identical(attr(m, "layout"), colnames(m)))
newdata <- data.frame(x = c(5, NA), g = factor(c("b", "b"), levels = c("a", "b")))
m2 <- glmnet_matrix(newdata, c("x", "g"), layout = attr(m, "layout"))
stopifnot(identical(colnames(m2), c("x", "gb", "gc")), all(m2[, "gc"] == 0), is.na(m2[2, "x"]))

# resolve_glmnet_family
stopifnot(resolve_glmnet_family(c(0, 3, 5)) == "poisson")
stopifnot(resolve_glmnet_family(c(0.5, 2)) == "gaussian")
stopifnot(resolve_glmnet_family(c("a", "b", NA)) == "binomial")
stopifnot(resolve_glmnet_family(factor(1:3)) == "multinomial")
stopifnot(resolve_glmnet_family(1:3, "gaussian") == "gaussian")

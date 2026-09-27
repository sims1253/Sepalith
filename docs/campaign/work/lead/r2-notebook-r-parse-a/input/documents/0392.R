fit.lasso <- function(
  expression.matrix,
  phenotype.vector,
  alpha = 1,
  nfolds = 10
) {
  x <- t(expression.matrix)
  fit <- glmnet::cv.glmnet(
    x,
    phenotype.vector,
    alpha = alpha,
    family = "binomial",
    nfolds = nfolds
  )

  return(fit)
}
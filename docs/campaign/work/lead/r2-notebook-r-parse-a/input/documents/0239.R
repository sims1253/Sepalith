Lincom_FtestAdj <- function(model_obj, L_value, denomdf_val = NULL) {
  if (sum(class(model_obj) == "glm") == 0) {
    stop("Model must be a glm object")
  }

  if (is.null(denomdf_val)) {
    denomdf_val <- model_obj$df.residual
  }

  if (denomdf_val <= 1) {
    stop("The degrees of freedom must be greater than 1")
  }

  robust_seHC3 <- sandwich::vcovHC(model_obj, type = "HC3")
  coefftestHC3 <- lmtest::coeftest(model_obj, vcov = robust_seHC3)

  robust_seHC2 <- sandwich::vcovHC(model_obj, type = "HC2")
  coefftestHC2 <- lmtest::coeftest(model_obj, vcov = robust_seHC2)

  beta_val <- matrix(
    coefftestHC3[, 1],
    nrow = length(coefftestHC3[, 1]),
    ncol = 1
  )
  Q_mat3 <- robust_seHC3
  Q_mat2 <- robust_seHC2
  numerdf_val <- dim(L_value)[1]
  new_fstat3 <- (t(beta_val) %*%
    t(L_value) %*%
    solve(L_value %*% Q_mat3 %*% t(L_value)) %*%
    L_value %*%
    beta_val) /
    Matrix::rankMatrix(L_value %*% Q_mat3 %*% t(L_value))[1]
  new_fstat2 <- (t(beta_val) %*%
    t(L_value) %*%
    solve(L_value %*% Q_mat2 %*% t(L_value)) %*%
    L_value %*%
    beta_val) /
    Matrix::rankMatrix(L_value %*% Q_mat2 %*% t(L_value))[1]
  #p-value for an f test
  avg_fstat <- mean(c(new_fstat2, new_fstat3))
  pval <- stats::pf(avg_fstat, numerdf_val, denomdf_val, lower.tail = FALSE)

  #c(numerdf_val, denomdf_val, avg_fstat, pval)

  return_val <- data.frame(
    Numerator_df = numerdf_val,
    Denominator_df = denomdf_val,
    SSAdjF_statistic = avg_fstat,
    SSAdjF_pvalue = pval
  )
  return_val
}
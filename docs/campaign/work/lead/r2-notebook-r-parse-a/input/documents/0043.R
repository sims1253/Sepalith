predict_posterior <- function(
  mod,
  dat,
  y,
  sim,
  inter = NULL,
  var_param,
  var_names
) {
  N <- dim(dat)[1]
  posterior_pred <- rstan::extract(mod)

  ysim <- matrix(NA, nrow = sim, ncol = N)

  ## Doing the matrix for the calculations
  matrix_model <- do.call(cbind, posterior_pred[var_param])
  data_model <- do.call(cbind, data[var_names])

  for (s in 1:sim) {
    if (is.null(inter)) {
      p <- plogis(matrix_model[s] %*% t(data_model))
      ysim[s, ] <- rbinom(N, size = 1, prob = p)
    } else {
      p <- plogis(
        matrix_model[s, inter] +
          matrix_model[s, -which(colnames(matrix_model) == inter)] %*%
            t(data_model)
      )
      ysim[s, ] <- rbinom(N, size = 1, prob = p)
    }
  }

  return(ysim)
}
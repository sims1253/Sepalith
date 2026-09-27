gradient_formula <- function(tau, theta, r_matrix, n) {
  initial_guess <- c(tau[1], tau[2], tau[3], tau[4])
  gradientvalue <- rootSolve::gradient(
    f = jointloglikelihood_tau_4param,
    x = initial_guess,
    theta = theta,
    r_matrix = r_matrix,
    n = n
  )
  return(gradientvalue)
}
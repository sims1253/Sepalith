fit_hmm <- function(
  formula,
  data,
  nstates = 2,
  seed = 42,
  maxiter = 100,
  tol = 1e-6
) {
  set.seed(seed)
  mf <- model.frame(formula, data)
  Y <- model.response(mf)
  X <- model.matrix(formula, data)
  n <- length(Y)
  p <- ncol(X)
  K <- nstates

  # Initialize parameters
  init_fit <- lm(Y ~ X - 1)
  init_beta <- coef(init_fit)
  init_sigma <- summary(init_fit)$sigma

  state_betas <- lapply(seq_len(K), function(k) init_beta + rnorm(p, 0, 0.1))
  state_sigmas <- rep(init_sigma, K)

  # Initialize transition matrix (diagonal dominance)
  trans_probs <- matrix(0.1, nrow = K, ncol = K)
  diag(trans_probs) <- 0.9
  trans_probs <- trans_probs / rowSums(trans_probs)

  # Initialize state probabilities
  init_probs <- rep(1 / K, K)

  # EM algorithm
  for (iter in seq_len(maxiter)) {
    old_params <- c(unlist(state_betas), state_sigmas, as.vector(trans_probs))

    # E-step: Forward-backward
    # Compute emission probabilities
    emit_probs <- matrix(0, nrow = n, ncol = K)
    for (k in seq_len(K)) {
      mu <- X %*% state_betas[[k]]
      emit_probs[, k] <- dnorm(Y, mu, state_sigmas[k])
    }

    # Forward pass
    alpha <- matrix(0, nrow = n, ncol = K)
    alpha[1, ] <- init_probs * emit_probs[1, ]
    alpha[1, ] <- alpha[1, ] / sum(alpha[1, ])

    for (t in 2:n) {
      for (j in seq_len(K)) {
        alpha[t, j] <- emit_probs[t, j] * sum(alpha[t - 1, ] * trans_probs[, j])
      }
      alpha[t, ] <- alpha[t, ] / sum(alpha[t, ])
    }

    # Backward pass
    beta <- matrix(0, nrow = n, ncol = K)
    beta[n, ] <- 1
    beta[n, ] <- beta[n, ] / sum(beta[n, ])

    for (t in (n - 1):1) {
      for (i in seq_len(K)) {
        beta[t, i] <- sum(
          trans_probs[i, ] * emit_probs[t + 1, ] * beta[t + 1, ]
        )
      }
      beta[t, ] <- beta[t, ] / sum(beta[t, ])
    }

    # Compute gamma (state probabilities) and xi (transition probabilities)
    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)

    xi <- array(0, dim = c(n - 1, K, K))
    for (t in 1:(n - 1)) {
      for (i in seq_len(K)) {
        for (j in seq_len(K)) {
          xi[t, i, j] <- alpha[t, i] *
            trans_probs[i, j] *
            emit_probs[t + 1, j] *
            beta[t + 1, j]
        }
      }
      xi[t, , ] <- xi[t, , ] / sum(xi[t, , ])
    }

    # M-step: Update parameters
    # Update transition probabilities
    trans_probs <- matrix(0, nrow = K, ncol = K)
    for (i in seq_len(K)) {
      for (j in seq_len(K)) {
        trans_probs[i, j] <- sum(xi[, i, j]) / sum(gamma[1:(n - 1), i])
      }
    }
    # Normalize rows
    trans_probs <- trans_probs / rowSums(trans_probs)

    # Update initial probabilities
    init_probs <- gamma[1, ]

    # Update state-specific parameters
    for (k in seq_len(K)) {
      wk <- gamma[, k]
      if (sum(wk) > 1e-10) {
        X_w <- X * wk
        state_betas[[k]] <- solve(t(X) %*% (X * wk)) %*% t(X_w) %*% Y
        resid_k <- Y - X %*% state_betas[[k]]
        state_sigmas[k] <- sqrt(sum(wk * resid_k^2) / sum(wk))
      }
    }

    # Check convergence
    new_params <- c(unlist(state_betas), state_sigmas, as.vector(trans_probs))
    if (max(abs(new_params - old_params)) < tol) break
  }

  # Viterbi algorithm for state decoding
  viterbi <- matrix(0, nrow = n, ncol = K)
  backptr <- matrix(0, nrow = n, ncol = K)

  viterbi[1, ] <- log(init_probs) + log(emit_probs[1, ])

  for (t in 2:n) {
    for (j in seq_len(K)) {
      max_val <- -Inf
      max_idx <- 1
      for (i in seq_len(K)) {
        val <- viterbi[t - 1, i] + log(trans_probs[i, j])
        if (val > max_val) {
          max_val <- val
          max_idx <- i
        }
      }
      viterbi[t, j] <- max_val + log(emit_probs[t, j])
      backptr[t, j] <- max_idx
    }
  }

  # Backtrack
  viterbi_states <- integer(n)
  viterbi_states[n] <- which.max(viterbi[n, ])
  for (t in (n - 1):1) {
    viterbi_states[t] <- backptr[t + 1, viterbi_states[t + 1]]
  }

  # Compute fitted values
  Y_hat <- numeric(n)
  for (i in seq_len(n)) {
    k <- viterbi_states[i]
    Y_hat[i] <- sum(X[i, ] * state_betas[[k]])
  }

  resid_raw <- Y - Y_hat
  arv <- make_arvind_resid(resid_raw, Y)

  result <- list(
    model_type = "Regime-Switching",
    fitted = Y_hat,
    residuals = resid_raw,
    theta = arv$theta,
    sigma = arv$sigma,
    shift = arv$shift,
    e_pos = arv$e_pos,
    negloglik = arv$negloglik,
    nstates = nstates,
    states = viterbi_states,
    trans_probs = trans_probs,
    state_betas = state_betas,
    state_sigmas = state_sigmas,
    n = n,
    p = p,
    X = X,
    Y = Y,
    formula = formula,
    data = data
  )
  class(result) <- "ArvindFit"
  return(result)
}
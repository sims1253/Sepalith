estimate_tau_x_paralell <- function(
  mat_allele_count,
  P,
  theta,
  fixed.n.at.tips,
  n.cores = NULL
) {
  # Generate all pairs where f > g
  pairs <- which(outer(1:P, 1:P, ">"), arr.ind = TRUE)

  # Set up cores
  if (is.null(n.cores)) {
    n.cores <- max(1, parallel::detectCores() - 1)
  }

  # Grab references to internal functions once, in the parent environment.
  # This ensures worker processes (both fork and PSOCK) can always find them,
  # regardless of whether the code is run as a script or as an installed package.
  .jll <- jointloglikelihood_tau_4param
  .grad <- gradient_formula

  # Worker function for a single pair
  compute_pair <- function(idx) {
    f <- pairs[idx, 1]
    g <- pairs[idx, 2]
    tip.allele.counts.submatrix <- cbind(
      mat_allele_count[g, ],
      mat_allele_count[f, ],
      mat_allele_count[P + 1, ]
    )
    eps <- 1e-10
    result <- optim(
      par = c(0, 0, 0, 0),
      fn = .jll,
      gr = .grad,
      method = "L-BFGS-B",
      lower = rep(eps, 4),
      upper = rep(Inf, 4),
      control = list(fnscale = -1),
      theta = theta,
      r_matrix = tip.allele.counts.submatrix,
      n = fixed.n.at.tips
    )
    return(list(f = f, g = g, x_est = result$par[4]))
  }

  message(sprintf("Running %d pairs on %d cores...", nrow(pairs), n.cores))

  if (.Platform$OS.type == "windows") {
    cl <- parallel::makeCluster(n.cores)
    on.exit(parallel::stopCluster(cl))

    # Export all variables needed by compute_pair, including the resolved
    # function references. envir = environment() captures .jll and .grad
    # along with the data variables, so workers get everything they need
    # without relying on package namespace lookup.
    parallel::clusterExport(
      cl,
      varlist = c(
        "pairs",
        "mat_allele_count",
        "P",
        "theta",
        "fixed.n.at.tips",
        ".jll",
        ".grad"
      ),
      envir = environment()
    )

    # rootSolve must be loaded on each worker because gradient_formula
    # calls rootSolve::gradient() internally.
    parallel::clusterEvalQ(cl, library(rootSolve))
  } else {
    # Mac / Linux: mclapply-style fork — child processes inherit the full
    # parent environment, so no export is needed.
    cl <- n.cores
  }

  # pblapply works with both a cluster object (Windows) and an integer (Mac/Linux)
  results <- pbapply::pblapply(seq_len(nrow(pairs)), compute_pair, cl = cl)

  # Fill symmetric matrix
  mat_est_tau_x <- matrix(nrow = P, ncol = P)
  for (res in results) {
    mat_est_tau_x[res$f, res$g] <- res$x_est
    mat_est_tau_x[res$g, res$f] <- res$x_est
  }

  return(mat_est_tau_x)
}
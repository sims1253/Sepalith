dwnorm2 <- function(
  x,
  kappa1 = 1,
  kappa2 = 1,
  kappa3 = 0,
  mu1 = 0,
  mu2 = 0,
  int.displ,
  log = FALSE
) {
  if (missing(int.displ)) {
    int.displ <- 3
  } else if (int.displ >= 5) {
    int.displ <- 5
  } else if (int.displ <= 1) {
    int.displ <- 1
  }
  displ <- floor(int.displ)
  omega.2pi.all <- expand.grid(-displ:displ, -displ:displ) * (2 * pi) # 2pi * integer displacements
  omega.2pi <- as.matrix(omega.2pi.all)

  if (any(c(kappa1, kappa2) < 0)) {
    stop("kappa1 and kappa2 must be non-negative")
  }
  if (any(mu1 < 0 | mu1 >= 2 * pi)) {
    mu1 <- prncp_reg(mu1)
  }
  if (any(mu2 < 0 | mu2 >= 2 * pi)) {
    mu2 <- prncp_reg(mu2)
  }
  if (
    (length(dim(x)) < 2 && length(x) != 2) ||
      (length(dim(x)) == 2 && tail(dim(x), 1) != 2) ||
      (length(dim(x)) > 2)
  ) {
    stop("x must either be a bivariate vector or a two-column matrix")
  }

  if (
    max(
      length(kappa1),
      length(kappa2),
      length(kappa3),
      length(mu1),
      length(mu2)
    ) >
      1
  ) {
    expanded <- expand_args(kappa1, kappa2, kappa3, mu1, mu2)
    kappa1 <- expanded[[1]]
    kappa2 <- expanded[[2]]
    kappa3 <- expanded[[3]]
    mu1 <- expanded[[4]]
    mu2 <- expanded[[5]]
  }

  par.mat <- rbind(kappa1, kappa2, kappa3, mu1, mu2)
  n_par <- ncol(par.mat)
  if (length(x) == 2) {
    x <- matrix(x, nrow = 1)
  }
  n_x <- nrow(x)

  if (all(kappa1 > 1e-10 | kappa2 > 1e-10)) {
    # regular wnorm2 density
    if (n_par == 1) {
      den <- c(dwnorm2_manyx_onepar(
        x,
        kappa1,
        kappa2,
        kappa3,
        mu1,
        mu2,
        omega.2pi
      ))
    } else if (n_x == 1) {
      den <- c(dwnorm2_onex_manypar(
        c(x),
        kappa1,
        kappa2,
        kappa3,
        mu1,
        mu2,
        omega.2pi
      ))
    } else {
      x_set <- 1:n_x
      par_set <- 1:n_par
      expndn_set <- expand_args(x_set, par_set)
      x_set <- expndn_set[[1]]
      par_set <- expndn_set[[2]]
      den <- c(dwnorm2_manyx_manypar(
        x[x_set, ],
        kappa1[par_set],
        kappa2[par_set],
        kappa3[par_set],
        mu1[par_set],
        mu2[par_set],
        omega.2pi
      ))
    }
  } else {
    # some can be uniform
    x_set <- 1:n_x
    par_set <- 1:n_par
    expndn_set <- expand_args(x_set, par_set)
    x_set <- expndn_set[[1]]
    par_set <- expndn_set[[2]]
    x_long <- x[x_set, , drop = FALSE]
    kappa1_long <- kappa1[par_set]
    kappa2_long <- kappa2[par_set]
    kappa3_long <- kappa3[par_set]
    mu1_long <- mu1[par_set]
    mu2_long <- mu2[par_set]
    n_x_final <- nrow(x_long)
    den <- rep(0, n_x_final)
    which_unif <- which(kappa1_long < 1e-10 & kappa2_long < 1e-10)
    n_unif <- length(which_unif)

    # browser()
    den[which_unif] <- 1 / (4 * pi^2)
    if (n_unif < n_x_final) {
      den[-which_unif] <- c(dwnorm2_manyx_manypar(
        x_long[-which_unif, , drop = FALSE],
        kappa1_long[-which_unif],
        kappa2_long[-which_unif],
        kappa3_long[-which_unif],
        mu1_long[-which_unif],
        mu2_long[-which_unif],
        omega.2pi
      ))
    }
  }

  if (log) {
    den <- log(den)
  }

  den
}
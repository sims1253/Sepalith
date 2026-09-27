caft_estimate <- function(
  otu.table,
  x,
  filter.thresh = 0.05,
  regularize = TRUE,
  n.cores = 1L
) {
  if (!is.matrix(otu.table)) {
    otu.table <- as.matrix(otu.table)
  }

  n.data <- NROW(otu.table)
  n.taxa <- NCOL(otu.table)

  if (missing(x) || is.null(x)) {
    stop("The full design matrix 'x' must be provided.", call. = FALSE)
  }
  x <- as.matrix(x)

  if (NROW(otu.table) != NROW(x)) {
    stop(
      " Number of samples not match between OTU table and covairates matrix!"
    )
  }
  if (anyNA(otu.table)) {
    stop(
      "otu.table contains missing values. Please remove or impute missing values before calling caft_estimate()."
    )
  }
  if (anyNA(x)) {
    stop(
      "The covariate matrix contains missing values. Please remove or impute missing values before calling caft_estimate()."
    )
  }

  lib.size <- rowSums(otu.table)
  if (any(lib.size <= 0)) {
    stop("All samples must have positive total library size.")
  }
  # center the covariates
  x.raw <- x
  x <- scale(x.raw, center = TRUE, scale = FALSE)
  x <- as.matrix(x)

  n.param <- NCOL(x)
  n.cores <- as.integer(n.cores)

  taxa.name <- colnames(otu.table)
  if (is.null(taxa.name)) {
    taxa.name <- paste0("taxon", seq_len(n.taxa))
    colnames(otu.table) <- taxa.name
  }
  x.name <- colnames(x.raw)
  if (is.null(x.name)) {
    x.name <- paste0("x", seq_len(n.param))
    colnames(x.raw) <- x.name
    colnames(x) <- x.name
  }

  # Relative abundance
  ra.all <- otu.table / lib.size
  # detection limits: Censored relative abundance: every row should same
  lim.ra <- matrix(1 / lib.size, nrow = n.data, ncol = n.taxa)

  #--------------------------
  # create survival data
  #--------------------------
  log_neg <- -log10(lim.ra)
  log_pos <- -log10(ra.all)

  t.star.all <- matrix(log_neg, nrow = n.data, ncol = n.taxa)
  idx <- (otu.table > 0)
  t.star.all[idx] <- log_pos[idx]

  t.star.all <- as.data.frame(t.star.all)
  colnames(t.star.all) <- taxa.name

  delta.all <- as.data.frame((otu.table > 0) * 1L)
  colnames(delta.all) <- taxa.name

  # fit_one_taxon(): one function for seq and parallel computing
  fit_one_taxon <- function(ii) {
    tstar <- t.star.all[[ii]]
    delta.1 <- delta.all[[ii]]

    if (sum(delta.1) <= n.data * filter.thresh) {
      return(list(
        beta = rep(NA_real_, n.param),
        skip_rare = 1L,
        skip_fail_fit = 0L,
        error_message = NA_character_
      ))
    }

    fit0.pen <- try(
      estimate.rank.aft(
        y = tstar,
        delta = delta.1,
        x = x,
        Gamma = NULL,
        Lambda = diag(n.param),
        Gamma.ginv = NULL,
        Lambda.ginv = diag(n.param),
        b = NULL,
        beta = NULL,
        test = TRUE,
        regularize = regularize,
        tol = 1e-12
      ),
      silent = TRUE
    )

    if (inherits(fit0.pen, "try-error")) {
      return(list(
        beta = rep(NA_real_, n.param),
        skip_rare = 0L,
        skip_fail_fit = 1L,
        error_message = as.character(fit0.pen)
      ))
    }

    list(
      beta = as.numeric(fit0.pen$beta),
      skip_rare = 0L,
      skip_fail_fit = 0L,
      error_message = NA_character_
    )
  }

  n.cores <- min(n.cores, n.taxa)

  if (n.cores > 1L) {
    cl <- parallel::makeCluster(n.cores, type = "PSOCK")
    on.exit(
      {
        try(parallel::stopCluster(cl), silent = TRUE)
        foreach::registerDoSEQ()
      },
      add = TRUE
    )
    doParallel::registerDoParallel(cl)

    res_phase1 <- foreach::foreach(
      ii = seq_len(n.taxa),
      .errorhandling = "pass"
    ) %dopar%
      {
        fit_one_taxon(ii)
      }
  } else {
    res_phase1 <- lapply(seq_len(n.taxa), fit_one_taxon)
  }

  # deal with errors possiblely from foreach
  res_phase1 <- lapply(res_phase1, function(z) {
    if (inherits(z, "error")) {
      list(
        beta = rep(NA_real_, n.param),
        skip_rare = 0L,
        skip_fail_fit = 1L,
        error_message = "Parallel unrestricted taxon-level fit failed."
      )
    } else {
      z
    }
  })

  beta.est <- do.call(rbind, lapply(res_phase1, `[[`, "beta"))
  beta.est <- as.data.frame(beta.est)
  colnames(beta.est) <- paste0("b", seq_len(n.param), ".est")
  rownames(beta.est) <- taxa.name

  skip.rare <- vapply(res_phase1, function(z) z$skip_rare, integer(1))
  skip.fail.rank.fit.pen <- vapply(
    res_phase1,
    function(z) z$skip_fail_fit,
    integer(1)
  )
  fit.error.messages <- vapply(
    res_phase1,
    function(z) z$error_message,
    character(1)
  )

  names(skip.rare) <- taxa.name
  names(skip.fail.rank.fit.pen) <- taxa.name
  names(fit.error.messages) <- taxa.name

  taxa.used <- taxa.name[
    skip.rare == 0L &
      skip.fail.rank.fit.pen == 0L &
      stats::complete.cases(beta.est)
  ]

  out <- list(
    call = match.call(),
    otu.table = otu.table,
    lib.size = lib.size,
    x.raw = x.raw,
    x = x,
    x.name = x.name,
    filter.thresh = filter.thresh,
    regularize = regularize,
    n.data = n.data,
    n.taxa = n.taxa,
    n.param = n.param,
    taxa.name = taxa.name,
    taxa.used = taxa.used,
    t.star.all = t.star.all,
    delta.all = delta.all,
    beta.est = beta.est,
    skip.otu = skip.rare,
    skip.rare = skip.rare,
    skip.fail.rank.fit.pen = skip.fail.rank.fit.pen,
    fit.error.messages = fit.error.messages
  )

  class(out) <- "caft_est"
  out
}
ctrml_fit <- function(
  hat,
  obs,
  agg_mat,
  agg_order,
  tew = "sum",
  features = "all",
  approach = "randomForest",
  params = NULL,
  tuning = NULL
) {
  # Check if 'agg_order' is provided
  if (missing(agg_order)) {
    cli_abort(
      "Argument {.arg agg_order} is missing, with no default.",
      call = NULL
    )
  }

  tmp <- cttools(agg_mat = agg_mat, agg_order = agg_order, tew = tew)
  strc_mat <- tmp$strc_mat

  id_bts <- c(rep(0, tmp$dim[["na"]]), rep(1, tmp$dim[["nb"]]))
  id_hfts <- c(rep(0, tmp$dim[["ks"]]), rep(1, tmp$dim[["m"]]))
  id_hfbts <- as.numeric(kronecker(id_bts, id_hfts))

  # block_sampling for the block tuning rtw option on mlr3
  block_sampling <- NULL

  if (missing(obs)) {
    cli_abort("Argument {.arg obs} is missing, with no default.", call = NULL)
  } else if (NCOL(obs) %% tmp$dim[["m"]] != 0) {
    cli_abort(
      'The number of columns of {.arg obs} must be a multiple of {tmp$dim[["m"]]}, but it is {NCOL(obs)}.',
      call = NULL
    )
  } else if (NROW(obs) != tmp$dim[["nb"]]) {
    cli_abort(
      '{.arg obs} must have {tmp$dim[["nb"]]} rows, but it has {NROW(obs)}.',
      call = NULL
    )
  } else {
    if (!grepl("mfh", features)) {
      obs <- t(obs)
    } else {
      obs <- matrix(
        as.vector(t(obs)),
        ncol = tmp$dim[["m"]] * tmp$dim[["nb"]]
      )
    }
  }

  if (missing(hat)) {
    cli_abort("Argument {.arg hat} is missing, with no default.", call = NULL)
  } else if (NCOL(hat) %% tmp$dim[["kt"]] != 0) {
    cli_abort(
      'The number of columns of {.arg hat} must be a multiple of {tmp$dim[["kt"]]}, but it is {NCOL(hat)}.',
      call = NULL
    )
  } else if (NROW(hat) != tmp$dim[["n"]]) {
    cli_abort(
      '{.arg hat} must have {tmp$dim[["n"]]} rows, but it has {NROW(hat)}.',
      call = NULL
    )
  } else {
    if (!grepl("mfh", features)) {
      hat <- input2rtw(hat, tmp$set)
    } else {
      h <- NCOL(hat) / tmp$dim[["kt"]]
      hat <- mat2hmat(hat, h = h, kset = tmp$set, n = tmp$dim[["n"]])
    }
  }

  switch(
    features,
    "mfh-hfbts" = {
      sel_mat <- as(id_hfbts, "sparseVector")
    },
    "mfh-hfts" = {
      sel_mat <- as(rep(id_hfts, tmp$dim[["n"]]), "sparseVector")
    },
    "mfh-bts" = {
      sel_mat <- as(rep(id_bts, each = tmp$dim[["kt"]]), "sparseVector")
    },
    "mfh-str" = {
      sel_mat <- 1 * (sel_mat != 0)
    },
    "mfh-str-hfbts" = {
      sel_mat <- 1 * (sel_mat != 0)
      sel_mat <- sel_mat +
        Matrix(
          rep(id_hfbts, tmp$dim[["nb"]] * tmp$dim[["m"]]),
          ncol = tmp$dim[["nb"]] * tmp$dim[["m"]],
          sparse = TRUE
        )
      sel_mat[sel_mat != 0] <- 1
    },
    "mfh-str-bts" = {
      sel_mat <- 1 * (sel_mat != 0)
      sel_mat <- sel_mat +
        Matrix(
          rep(
            rep(id_bts, each = tmp$dim[["kt"]]),
            tmp$dim[["nb"]] * tmp$dim[["m"]]
          ),
          ncol = tmp$dim[["nb"]] * tmp$dim[["m"]],
          sparse = TRUE
        )
      sel_mat[sel_mat != 0] <- 1
    },
    "mfh-all" = {
      sel_mat <- 1
    },
    "all" = {
      sel_mat <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    "compact" = {
      pos <- seq(
        tmp$dim[["na"]],
        by = tmp$dim[["n"]],
        length.out = tmp$dim[["p"]]
      )
      sel_mat <- Matrix::bandSparse(
        tmp$dim[["nb"]],
        tmp$dim[["n"]] * tmp$dim[["p"]],
        pos
      )
      sel_mat <- 1 * t(sel_mat)
      sel_mat[1:tmp$dim[["n"]], ] <- 1
      block_sampling <- tmp$dim[["m"]]
    },
    {
      cli_abort(
        '{.arg features} = {.val {features}} is not a valid option.',
        call = NULL
      )
    }
  )
  attr(sel_mat, "sel_method") <- features

  # Remove NA variables from sel_mat
  na_var <- colSums(is.na(hat)) >= 0.75 * NROW(hat)
  if (any(na_var)) {
    if (NCOL(sel_mat) == 1) {
      if (length(sel_mat) == 1) {
        sel_mat <- rep(sel_mat, NCOL(hat))
      }
      sel_mat[na_var] <- 0
      sel_mat <- as(sel_mat, "sparseVector")
    } else {
      sel_mat[na_var, ] <- 0
    }
  }

  obj <- rml(
    base = NULL,
    hat = hat,
    obs = obs,
    sel_mat = sel_mat,
    approach = approach,
    params = params,
    fit = NULL,
    tuning = tuning,
    block_sampling = block_sampling
  )

  obj <- new_rml_fit(
    fit = obj$fit,
    agg_mat = agg_mat,
    agg_order = agg_order,
    tew = tew,
    sel_mat = obj$sel_mat,
    approach = approach,
    framework = "cross-temporal",
    features = features,
    features_size = NCOL(hat),
    sample_size = NROW(hat),
    block_sampling = block_sampling
  )
  return(obj)
}
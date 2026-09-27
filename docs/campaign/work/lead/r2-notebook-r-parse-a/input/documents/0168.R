boot_dr <- function(
  obj,
  data,
  R = 200L,
  level = 0.95,
  pw = FALSE,
  seed = NULL,
  parallel = FALSE
) {
  stopifnot(inherits(obj, "extract"))
  stopifnot(is.data.frame(data))
  stopifnot(level > 0, level < 1)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # ---- recover column names and settings from the stored call ---------------
  call_list <- as.list(obj$call)
  index_col <- if (!is.null(call_list$index_col)) {
    as.character(call_list$index_col)
  } else {
    "index"
  }
  n_col <- if (!is.null(call_list$n_col)) {
    as.character(call_list$n_col)
  } else {
    NULL
  }
  n_dim <- obj$settings$n_dim # 1 or 2

  if (!index_col %in% names(data)) {
    stop("Column '", index_col, "' not found in data.")
  }
  if (!is.null(n_col) && !n_col %in% names(data)) {
    stop("Column '", n_col, "' not found in data.")
  }

  # ---- argument list for replaying the call ---------------------------------
  # as.list(call)[-1] drops the function name; we replace $data each iteration
  boot_args <- as.list(obj$call)[-1]

  # ---- prepare binomial draw parameters ------------------------------------
  idx <- data[[index_col]]
  ns <- if (!is.null(n_col)) {
    as.numeric(data[[n_col]])
  } else {
    rep(1000, nrow(data))
  }

  pct_scale <- max(idx, na.rm = TRUE) > 1
  prop <- if (pct_scale) idx / 100 else idx
  prop <- pmin(pmax(prop, 0), 1)

  one_rep <- function(r) {
    sim_y <- rbinom(length(prop), size = round(ns), prob = prop)
    sim_index <- sim_y / ns
    if (pct_scale) {
      sim_index <- sim_index * 100
    }

    d <- data
    d[[index_col]] <- sim_index

    the_args <- boot_args
    the_args$data <- d

    res <- tryCatch(do.call(extract, the_args), error = function(e) NULL)
    if (is.null(res)) {
      return(NULL)
    }
    if (n_dim == 2L) {
      list(mood = res$mood, mood_dim2 = res$mood_dim2)
    } else {
      res$mood
    }
  }

  # ---- run replications -----------------------------------------------------
  if (parallel && requireNamespace("parallel", quietly = TRUE)) {
    nc <- max(1L, parallel::detectCores() - 1L)
    cl <- parallel::makeCluster(nc)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterExport(
      cl,
      c("data", "prop", "ns", "index_col", "pct_scale", "boot_args", "n_dim"),
      envir = environment()
    )
    parallel::clusterEvalQ(cl, library(DyadRatios))
    results <- parallel::parLapply(cl, seq_len(R), one_rep)
  } else {
    results <- lapply(seq_len(R), one_rep)
  }

  ok <- !sapply(results, is.null)
  results <- results[ok]
  if (length(results) == 0L) {
    stop("All bootstrap replications failed.")
  }

  # ---- build sample matrices ------------------------------------------------
  alpha <- (1 - level) / 2
  if (n_dim == 2L) {
    mood_mat <- do.call(cbind, lapply(results, `[[`, "mood"))
    mood_mat_dim2 <- do.call(cbind, lapply(results, `[[`, "mood_dim2"))
  } else {
    mood_mat <- do.call(cbind, results)
  }

  # ---- assemble summary data frame ------------------------------------------
  estimates <- cbind(
    obj$periods,
    data.frame(
      mood = obj$mood,
      lower = apply(mood_mat, 1, quantile, probs = alpha, na.rm = TRUE),
      upper = apply(mood_mat, 1, quantile, probs = 1 - alpha, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  )
  if (n_dim == 2L) {
    estimates$mood_dim2 <- obj$mood_dim2
    estimates$lower_dim2 <- apply(
      mood_mat_dim2,
      1,
      quantile,
      probs = alpha,
      na.rm = TRUE
    )
    estimates$upper_dim2 <- apply(
      mood_mat_dim2,
      1,
      quantile,
      probs = 1 - alpha,
      na.rm = TRUE
    )
  }

  # ---- assemble return list -------------------------------------------------
  out <- list(estimates = estimates, samples = mood_mat)
  if (n_dim == 2L) {
    out$samples_dim2 <- mood_mat_dim2
  }

  # ---- calculate optional pairwise differences ------------------------------
  if (pw) {
    period <- estimates$period
    combs <- combn(nrow(mood_mat), 2)
    D <- matrix(0, nrow = nrow(mood_mat), ncol = ncol(combs))
    D[cbind(combs[1, ], seq_len(ncol(combs)))] <- -1
    D[cbind(combs[2, ], seq_len(ncol(combs)))] <- 1
    diffs <- t(mood_mat) %*% D
    p_diff <- apply(diffs, 2, function(x) mean(x > 0))
    p_diff <- ifelse(p_diff < 0.5, 1 - p_diff, p_diff)
    diff <- matrix(estimates$mood, nrow = 1) %*% D
    out$pw <- data.frame(
      p1 = period[combs[1, ]],
      p2 = period[combs[2, ]],
      diff = c(diff),
      p_diff = p_diff
    )
  }

  attr(out, "R") <- sum(ok)
  attr(out, "level") <- level
  attr(out, "agg_interval") <- obj$settings$agg_interval
  attr(out, "n_dim") <- n_dim
  class(out) <- "boot_dr"
  out
}
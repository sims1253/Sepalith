did_multiplegt_main_smaller <- function(
  df,
  boot_weight_col,
  outcome,
  group,
  time,
  treatment,
  effects,
  placebo,
  ci_level,
  switchers,
  trends_nonparam,
  weight,
  controls,
  dont_drop_larger_lower,
  drop_if_d_miss_before_first_switch,
  cluster,
  same_switchers,
  same_switchers_pl,
  only_never_switchers,
  effects_equal,
  save_results,
  normalized,
  predict_het,
  trends_lin,
  less_conservative_se,
  continuous
) {
  # --- Build the effective weight column -----------------------------------
  # If the user did not supply weights, the bootstrap multiplicity column
  # *becomes* the weight. If they did supply weights, we multiply the two
  # so the bootstrap acts as a duplication on top of the original weights.
  boot_w <- df[[boot_weight_col]]
  if (is.null(boot_w)) {
    stop(
      "did_multiplegt_main_smaller: bootstrap weight column '",
      boot_weight_col,
      "' missing from the slim dataset."
    )
  }
  boot_w[is.na(boot_w)] <- 0

  if (is.null(weight) || identical(weight, "")) {
    df[[".__did_boot_weight_XX"]] <- as.numeric(boot_w)
    weight_used <- ".__did_boot_weight_XX"
  } else {
    user_w <- df[[weight]]
    user_w[is.na(user_w)] <- 0
    df[[weight]] <- as.numeric(user_w) * as.numeric(boot_w)
    weight_used <- weight
  }

  # Drop rows whose effective weight is zero -- those units were not drawn
  # in this bootstrap replicate. This keeps the slim dataset truly slim and
  # avoids feeding zero-weight rows to the polars pipeline.
  keep <- df[[weight_used]] > 0
  keep[is.na(keep)] <- FALSE
  if (!all(keep)) {
    df <- df[keep, , drop = FALSE]
  }
  df[[boot_weight_col]] <- NULL

  # Hand off to the existing engine. Memory reclaim is handled by the
  # surrounding subprocess: callr::r() exits and the OS reclaims the
  # polars buffers.
  df_est <- did_multiplegt_main(
    df = df,
    outcome = outcome,
    group = group,
    time = time,
    treatment = treatment,
    effects = effects,
    placebo = placebo,
    ci_level = ci_level,
    switchers = switchers,
    trends_nonparam = trends_nonparam,
    weight = weight_used,
    controls = controls,
    dont_drop_larger_lower = dont_drop_larger_lower,
    drop_if_d_miss_before_first_switch = drop_if_d_miss_before_first_switch,
    cluster = cluster,
    same_switchers = same_switchers,
    same_switchers_pl = same_switchers_pl,
    only_never_switchers = only_never_switchers,
    effects_equal = effects_equal,
    save_results = save_results,
    normalized = normalized,
    predict_het = predict_het,
    trends_lin = trends_lin,
    less_conservative_se = less_conservative_se,
    continuous = continuous
  )

  res <- df_est$did_multiplegt_dyn

  # Return ONLY the numeric vectors the bootstrap aggregator needs.
  # Everything else (vcov matrices, panel objects, polars frames) is allowed
  # to die with the subprocess.
  out <- list(
    Effects = if (!is.null(res$Effects)) {
      as.numeric(res$Effects[, 1])
    } else {
      numeric(0)
    },
    ATE = if (!is.null(res$ATE)) as.numeric(res$ATE[1]) else NA_real_,
    Placebos = if (!is.null(res$Placebos)) {
      as.numeric(res$Placebos[, 1])
    } else {
      numeric(0)
    }
  )
  out
}
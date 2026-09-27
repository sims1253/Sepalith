validate_engine_specific_values <- function(engine, resolved_engine_args) {
  is_scalar_integerish <- function(x) {
    is.numeric(x) && length(x) == 1L && is.finite(x) && abs(x - round(x)) < 1e-8
  }

  if (engine == "AGHQ") {
    if ("outer_derivative_method" %in% names(resolved_engine_args)) {
      method <- resolved_engine_args$outer_derivative_method
      valid <- c("tmb", "finite_difference")
      if (
        !is.character(method) || length(method) != 1L || !(method %in% valid)
      ) {
        stop(
          "`outer_derivative_method` must be one of: ",
          paste(valid, collapse = ", "),
          call. = FALSE
        )
      }
      if (identical(method, "finite_difference")) {
        opt <- resolved_engine_args$optimizer
        if (is.null(opt) || !identical(opt, "nlminb")) {
          stop(
            "`outer_derivative_method = \"finite_difference\"` for AGHQ requires ",
            "`engine.args = list(optimizer = \"nlminb\")`.",
            call. = FALSE
          )
        }
      }
    }
    if ("aghq_k" %in% names(resolved_engine_args)) {
      k <- resolved_engine_args$aghq_k
      if (!is_scalar_integerish(k) || k < 1) {
        stop("`aghq_k` must be an integer-like scalar >= 1.", call. = FALSE)
      }
      resolved_engine_args$aghq_k <- as.integer(round(k))
    }
    if ("optimizer" %in% names(resolved_engine_args)) {
      opt <- resolved_engine_args$optimizer
      if (!is.character(opt) || length(opt) != 1L || !nzchar(opt)) {
        stop("`optimizer` must be a non-empty character scalar.", call. = FALSE)
      }
    }
  }

  if (engine == "TMB") {
    if ("outer_derivative_method" %in% names(resolved_engine_args)) {
      method <- resolved_engine_args$outer_derivative_method
      valid <- c("tmb", "finite_difference")
      if (
        !is.character(method) || length(method) != 1L || !(method %in% valid)
      ) {
        stop(
          "`outer_derivative_method` must be one of: ",
          paste(valid, collapse = ", "),
          call. = FALSE
        )
      }
    }
    if ("iterations" %in% names(resolved_engine_args)) {
      it <- resolved_engine_args$iterations
      if (!is_scalar_integerish(it) || it < 1) {
        stop("`iterations` must be an integer-like scalar >= 1.", call. = FALSE)
      }
      resolved_engine_args$iterations <- as.integer(round(it))
    }
    if ("hess_control_parscale" %in% names(resolved_engine_args)) {
      ps <- resolved_engine_args$hess_control_parscale
      if (!(is.null(ps) || is.numeric(ps))) {
        stop("`hess_control_parscale` must be numeric or NULL.", call. = FALSE)
      }
    }
    if ("hess_control_ndeps" %in% names(resolved_engine_args)) {
      nd <- resolved_engine_args$hess_control_ndeps
      if (!is.numeric(nd) || length(nd) != 1L || !is.finite(nd) || nd <= 0) {
        stop(
          "`hess_control_ndeps` must be a numeric scalar > 0.",
          call. = FALSE
        )
      }
    }
  }

  if (engine == "MCMC") {
    reserved <- intersect(
      names(resolved_engine_args),
      c(
        "data",
        "priors",
        "family",
        "link",
        "engine",
        "time_varying_betas",
        "fixed_effect_betas",
        "engine.args",
        "aghq_k",
        "field",
        "iid",
        "silent",
        "starting_values",
        "optimizer",
        "verbose"
      )
    )
    if (length(reserved)) {
      stop(
        "The following MCMC `engine.args` key(s) must be supplied as top-level arguments instead: ",
        paste(reserved, collapse = ", "),
        ".",
        call. = FALSE
      )
    }
    resolved_engine_args <- validate_mcmc_engine_args_values(
      resolved_engine_args
    )
  }

  resolved_engine_args
}
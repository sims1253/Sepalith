#' Typical values of model covariates
#'
#' Returns a one-row data frame holding the median of every numeric column and
#' the most frequent level of every factor or character column.
#'
#' @param data A data frame containing the covariates.
#' @param exclude Character vector of columns to leave out.
#' @return A one-row data frame.
#' @keywords internal
typicalValues <- function(data, exclude = character()) {
  keep <- setdiff(names(data), exclude)
  out <- vector("list", length(keep))
  names(out) <- keep
  for (col in keep) {
    x <- data[[col]]
    if (is.numeric(x)) {
      out[[col]] <- stats::median(x, na.rm = TRUE)
    } else if (is.factor(x)) {
      counts <- table(x)
      out[[col]] <- factor(names(counts)[which.max(counts)], levels = levels(x))
    } else {
      counts <- table(as.character(x))
      out[[col]] <- names(counts)[which.max(counts)]
    }
  }
  return(as.data.frame(out, stringsAsFactors = FALSE))
}

#' Covariate behind a smooth term
#'
#' Looks up the first covariate of a smooth by its label in a fitted GAM.
#'
#' @param fit A fitted `gam` object.
#' @param term Label of a smooth term, such as `"s(age)"`.
#' @return A single string.
#' @keywords internal
smoothTermVar <- function(fit, term) {
  labels <- vapply(fit$smooth, function(sm) sm$label, character(1))
  idx <- match(term, labels)
  if (is.na(idx)) {
    stop(
      sprintf("Smooth `%s` not found; available: %s.", term, paste(labels, collapse = ", ")),
      call. = FALSE
    )
  }
  return(fit$smooth[[idx]]$term[1L])
}

#' Grid for a partial effect
#'
#' Spans the observed range of one covariate and holds every other covariate
#' at a typical value.
#'
#' @param data A data frame containing the covariates.
#' @param var Name of the covariate to vary.
#' @param n Number of grid points.
#' @return A data frame with `n` rows.
#' @keywords internal
partialEffectGrid <- function(data, var, n = 100) {
  x <- data[[var]]
  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric to draw a partial effect.", var), call. = FALSE)
  }
  rng <- range(x, na.rm = TRUE)
  grid <- typicalValues(data, exclude = var)
  grid <- grid[rep(1L, n), , drop = FALSE]
  grid[[var]] <- seq(rng[1L], rng[2L], length.out = n)
  rownames(grid) <- NULL
  return(grid)
}

#' Partial effect of a smooth term
#'
#' Predicts the centred contribution of one smooth over a grid of its
#' covariate, with a pointwise confidence band.
#'
#' @param fit A fitted `gam` object.
#' @param term Label of a smooth term, such as `"s(age)"`.
#' @param data A data frame containing the covariates.
#' @param n Number of grid points.
#' @param level Confidence level for the band.
#' @return A tibble with columns `x`, `effect`, `lower` and `upper`.
#' @export
partialEffectData <- function(fit, term, data, n = 100, level = 0.95) {
  var <- smoothTermVar(fit, term)
  grid <- partialEffectGrid(data, var, n = n)
  pred <- mgcv::predict.gam(fit, newdata = grid, type = "terms", terms = term, se.fit = TRUE)
  crit <- stats::qnorm(1 - (1 - level) / 2)
  effect <- pred$fit[, term]
  se <- pred$se.fit[, term]
  out <- tibble::tibble(
    term = term,
    x = grid[[var]],
    effect = effect,
    lower = effect - crit * se,
    upper = effect + crit * se
  )
  return(out)
}

#' Plot the partial effect of a smooth term
#'
#' Draws the estimated smooth with its confidence band and a rug of observed
#' covariate values.
#'
#' @param fit A fitted `gam` object.
#' @param term Label of a smooth term, such as `"s(age)"`.
#' @param data A data frame containing the covariates.
#' @param level Confidence level for the band.
#' @param rug Whether to draw a rug of observed values.
#' @param residuals Whether to overlay partial residuals; `data` must then be
#'   the data the model was fitted to.
#' @return A ggplot object.
#' @importFrom rlang .data
#' @export
plotPartialEffect <- function(fit, term, data, level = 0.95, rug = TRUE,
                              residuals = FALSE) {
  effects <- partialEffectData(fit, term, data, level = level)
  var <- smoothTermVar(fit, term)
  p <- ggplot2::ggplot(effects, ggplot2::aes(x = .data$x, y = .data$effect)) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$lower, ymax = .data$upper),
      fill = "grey80",
      alpha = 0.6
    ) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::labs(x = var, y = paste("Partial effect of", term))
  if (rug) {
    observed <- data.frame(x = data[[var]])
    p <- p + ggplot2::geom_rug(
      data = observed,
      ggplot2::aes(x = .data$x),
      inherit.aes = FALSE,
      alpha = 0.3
    )
  }
  if (residuals) {
    if (nrow(data) != length(fit$residuals)) {
      stop("Partial residuals need the data used to fit the model.", call. = FALSE)
    }
    contribution <- mgcv::predict.gam(fit, newdata = data, type = "terms", terms = term)
    partial <- data.frame(
      x = data[[var]],
      y = stats::residuals(fit, type = "working") + contribution[, term]
    )
    p <- p + ggplot2::geom_point(
      data = partial,
      ggplot2::aes(x = .data$x, y = .data$y),
      inherit.aes = FALSE,
      size = 0.8,
      alpha = 0.4
    )
  }
  return(p)
}

#' Plot every univariate smooth in a GAM
#'
#' Builds one partial-effect plot per one-dimensional smooth; tensor products
#' are skipped because they need a surface plot.
#'
#' @param fit A fitted `gam` object.
#' @param data A data frame containing the covariates.
#' @param level Confidence level for the band.
#' @return A named list of ggplot objects.
#' @export
plotAllPartials <- function(fit, data, level = 0.95) {
  stopifnot(inherits(fit, "gam"))
  plots <- list()
  for (sm in fit$smooth) {
    if (sm$dim > 1L) {
      message("Skipping multi-dimensional smooth ", sm$label)
      next
    }
    if (!is.null(sm$by.level)) {
      message("Skipping factor-by smooth ", sm$label)
      next
    }
    plots[[sm$label]] <- plotPartialEffect(fit, sm$label, data, level = level)
  }
  return(plots)
}

#' Range of partial effects
#'
#' Summarises how much each smooth moves the linear predictor across the
#' observed covariate range, a quick way to rank smooths by importance.
#'
#' @param fit A fitted `gam` object.
#' @param data A data frame containing the covariates.
#' @return A tibble with one row per univariate smooth, largest range first.
#' @importFrom rlang .data
#' @export
partialEffectRange <- function(fit, data) {
  stopifnot(inherits(fit, "gam"))
  labels <- fit$smooth %>%
    purrr::keep(function(sm) sm$dim == 1L && is.null(sm$by.level)) %>%
    purrr::map_chr("label")
  ranges <- labels %>%
    purrr::map_dfr(function(term) {
      effects <- partialEffectData(fit, term, data)
      tibble::tibble(
        term = term,
        min = min(effects$effect),
        max = max(effects$effect),
        range = max(effects$effect) - min(effects$effect)
      )
    }) %>%
    dplyr::arrange(dplyr::desc(.data$range))
  return(ranges)
}

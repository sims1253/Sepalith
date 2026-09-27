summary.BayesPET_predtime <- function(object, ...) {
  if (!is.list(object) || is.null(object$prediction)) {
    stop("Invalid 'BayesPET_predtime' object: missing component 'prediction'.",
         call. = FALSE)
  }

  pred <- object$prediction
  S <- length(pred)
  n_inf <- sum(!is.finite(pred))

  qs <- if (length(pred) > 0L) {
    stats::quantile(pred,
                    probs = c(0.25, 0.5, 0.75),
                    na.rm = FALSE,
                    names = TRUE,
                    type = 7)
  } else {
    stats::setNames(rep(NA_real_, 3), c("25%", "50%", "75%"))
  }

  out <- list(
    S = S,
    n_infinite = n_inf,
    prob_not_reached = if (S > 0) n_inf / S else NA_real_,
    q25 = unname(qs[[1]]),
    median = unname(qs[[2]]),
    q75 = unname(qs[[3]]),
    call = object$call
  )

  class(out) <- "summary.BayesPET_predtime"
  out
}
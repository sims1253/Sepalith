predict.modelFits <- function(
  object,
  doses = NULL,
  probability_scale = attr(object, "probability_scale"),
  ...
) {
  checkmate::assert_double(
    doses,
    lower = 0,
    any.missing = FALSE,
    unique = TRUE,
    sorted = TRUE,
    finite = TRUE,
    null.ok = TRUE
  )
  checkmate::assert_flag(probability_scale)

  model_fits <- object
  model_names <- names(model_fits)

  warning_displayed <- FALSE
  withCallingHandlers(
    expr = {
      predictions <- lapply(
        model_fits[model_names != "avgFit"],
        predictModelFit,
        doses = doses
      )

      if ("avgFit" %in% model_names) {
        preds_avg_fit <- predictAvgFit(model_fits, doses = doses) # predictAvgFit calls predict.modelFits !!
        predictions <- c(list(avgFit = preds_avg_fit), predictions)
      }
    },

    warning = function(w) {
      if (
        grepl(
          "The specified range exceeds the bounds of the original dose range.",
          conditionMessage(w)
        )
      ) {
        if (warning_displayed) {
          invokeRestart("muffleWarning")
        } else {
          warning_displayed <<- TRUE
        }
      }
    }
  )

  if (probability_scale) {
    predictions <- lapply(predictions, RBesT::inv_logit)
  }

  attr(predictions, "doses") <- doses

  return(predictions)
}
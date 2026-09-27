find_optimal_threshold_dia <- function(
  prob_positive,
  y_true,
  type = c("f1", "youden"),
  pos_class,
  neg_class
) {
  type <- match.arg(type)
  thresholds <- unique(sort(c(0, prob_positive, 1)))
  thresholds <- thresholds[thresholds > 0 & thresholds < 1]
  if (length(thresholds) == 0) {
    thresholds <- 0.5
  } # Fallback if no unique internal probabilities

  best_score <- -Inf
  best_threshold <- 0.5
  y_true_factor <- factor(y_true, levels = c(neg_class, pos_class))

  for (t in thresholds) {
    y_pred_class <- factor(
      base::ifelse(prob_positive >= t, pos_class, neg_class),
      levels = c(neg_class, pos_class)
    )
    cm <- suppressWarnings(caret::confusionMatrix(
      y_pred_class,
      y_true_factor,
      positive = pos_class
    ))
    current_score <- NA
    if (type == "f1") {
      current_score <- cm$byClass["F1"]
    } else if (type == "youden") {
      current_score <- cm$byClass["Sensitivity"] + cm$byClass["Specificity"] - 1
    }
    if (!is.na(current_score) && current_score > best_score) {
      best_score <- current_score
      best_threshold <- t
    }
  }
  return(best_threshold)
}
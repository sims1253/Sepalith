#' Build a confusion matrix
#'
#' Cross-tabulates observed and predicted class labels using a shared set of
#' levels, so that empty classes still appear as zero rows and columns.
#'
#' @param truth A factor or character vector of observed classes.
#' @param estimate A factor or character vector of predicted classes.
#' @param levels Optional character vector of class levels, in display order.
#' @return An object of class `fw_conf_mat` wrapping an integer table.
#' @family confusion
#' @export
#' @examples
#' conf_mat_build(c("a", "b", "a"), c("a", "a", "a"))
conf_mat_build <- function(truth, estimate, levels = NULL) {
  if (length(truth) != length(estimate)) {
    stop("`truth` and `estimate` must have the same length.", call. = FALSE)
  }
  if (is.null(levels)) {
    levels <- sort(union(as.character(truth), as.character(estimate)))
  }
  tab <- table(
    Prediction = factor(estimate, levels = levels),
    Truth = factor(truth, levels = levels)
  )
  structure(list(table = tab, levels = levels), class = "fw_conf_mat")
}

#' Per-class true and false counts
#'
#' Extracts true positive, false positive and false negative counts for every
#' class in a confusion matrix.
#'
#' @param cm An object of class `fw_conf_mat`.
#' @return A data frame with one row per class.
#' @keywords internal
class_counts <- function(cm) {
  tab <- unclass(cm$table)
  tp <- diag(tab)
  data.frame(
    class = cm$levels,
    tp = as.integer(tp),
    fp = as.integer(rowSums(tab) - tp),
    fn = as.integer(colSums(tab) - tp),
    stringsAsFactors = FALSE
  )
}

#' Average a per-class statistic
#'
#' Combines per-class values with macro, weighted, or micro averaging.
#'
#' @param num Numerators, one per class.
#' @param den Denominators, one per class.
#' @param support Number of true observations per class.
#' @param average One of `"macro"`, `"weighted"` or `"micro"`.
#' @return A single numeric value.
#' @keywords internal
average_by_class <- function(num, den, support, average = c("macro", "weighted", "micro")) {
  average <- match.arg(average)
  per_class <- ifelse(den > 0, num / den, NA_real_)
  switch(average,
    macro = mean(per_class, na.rm = TRUE),
    weighted = stats::weighted.mean(per_class, support, na.rm = TRUE),
    micro = sum(num) / sum(den)
  )
}

#' Precision from a confusion matrix
#'
#' Computes the share of predicted positives that are correct, averaged
#' across classes.
#'
#' @param cm An object of class `fw_conf_mat`.
#' @param average One of `"macro"`, `"weighted"` or `"micro"`.
#' @return A single numeric value.
#' @family confusion
#' @export
cm_precision <- function(cm, average = c("macro", "weighted", "micro")) {
  counts <- class_counts(cm)
  support <- counts$tp + counts$fn
  average_by_class(counts$tp, counts$tp + counts$fp, support, match.arg(average))
}

#' Recall from a confusion matrix
#'
#' Computes the share of true positives that were recovered, averaged across
#' classes.
#'
#' @param cm An object of class `fw_conf_mat`.
#' @param average One of `"macro"`, `"weighted"` or `"micro"`.
#' @return A single numeric value.
#' @family confusion
#' @export
cm_recall <- function(cm, average = c("macro", "weighted", "micro")) {
  counts <- class_counts(cm)
  support <- counts$tp + counts$fn
  average_by_class(counts$tp, counts$tp + counts$fn, support, match.arg(average))
}

#' F1 score from a confusion matrix
#'
#' Computes the harmonic mean of precision and recall for each class and then
#' averages across classes.
#'
#' @param cm An object of class `fw_conf_mat`.
#' @param average One of `"macro"`, `"weighted"` or `"micro"`.
#' @return A single numeric value.
#' @family confusion
#' @export
cm_f1 <- function(cm, average = c("macro", "weighted", "micro")) {
  counts <- class_counts(cm)
  support <- counts$tp + counts$fn
  # F1 = 2TP / (2TP + FP + FN), which avoids dividing precision by recall
  average_by_class(2 * counts$tp, 2 * counts$tp + counts$fp + counts$fn, support,
    match.arg(average))
}

#' Overall accuracy from a confusion matrix
#'
#' Computes the share of all observations on the diagonal.
#'
#' @param cm An object of class `fw_conf_mat`.
#' @return A single numeric value.
#' @family confusion
#' @export
cm_accuracy <- function(cm) {
  tab <- unclass(cm$table)
  total <- sum(tab)
  if (total == 0) {
    return(NA_real_)
  }
  sum(diag(tab)) / total
}

#' Print a confusion matrix
#'
#' Prints the cross-tabulation followed by overall accuracy.
#'
#' @param x An object of class `fw_conf_mat`.
#' @param ... Passed to [print.table()].
#' @return `x`, invisibly.
#' @export
print.fw_conf_mat <- function(x, ...) {
  cat("Confusion matrix (rows = prediction, columns = truth)\n\n")
  print(x$table, ...)
  cat(sprintf("\nAccuracy: %.3f\n", cm_accuracy(x)))
  invisible(x)
}

#' Summarise a confusion matrix
#'
#' Returns per-class precision, recall and F1 alongside class support.
#'
#' @param object An object of class `fw_conf_mat`.
#' @param ... Ignored.
#' @return A data frame with one row per class.
#' @export
summary.fw_conf_mat <- function(object, ...) {
  counts <- class_counts(object)
  safe_div <- function(a, b) ifelse(b > 0, a / b, NA_real_)
  counts$support <- counts$tp + counts$fn
  counts$precision <- safe_div(counts$tp, counts$tp + counts$fp)
  counts$recall <- safe_div(counts$tp, counts$tp + counts$fn)
  counts$f1 <- safe_div(2 * counts$tp, 2 * counts$tp + counts$fp + counts$fn)
  counts[c("class", "support", "precision", "recall", "f1")]
}

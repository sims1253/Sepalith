#' Comprehensive multiple-choice item analysis
#'
#' Runs the full classical test theory item analysis on a multiple-choice
#' response matrix and returns a tidy `mcq_analysis` object containing
#' per-item difficulty, discrimination (both point-biserial and the
#' upper-lower 27 percent index), distractor efficiency, and the full
#' per-option distractor analysis. The returned object has dedicated
#' `print()`, `plot()`, and `apa_table()` methods.
#'
#' @param responses A matrix or data frame of student responses, with
#'   students in rows and items in columns.
#' @param key A vector of correct answers with length equal to the number
#'   of items.
#' @param options Optional character vector listing all possible response
#'   options. If `NULL` (default), inferred from the data.
#' @param min_proportion Minimum proportion of examinees selecting a
#'   distractor for it to be considered functioning when computing
#'   distractor efficiency. Default 0.05.
#'
#' @return An object of class `mcq_analysis` (a list) with components:
#'   \describe{
#'     \item{`items`}{Data frame with one row per item summarizing
#'       difficulty, point-biserial, discrimination index, and
#'       distractor efficiency.}
#'     \item{`distractors`}{Data frame with full per-option distractor
#'       analysis (one row per item-option combination).}
#'     \item{`total_scores`}{Numeric vector of total test scores, one
#'       per student.}
#'     \item{`n_students`}{Number of students.}
#'     \item{`n_items`}{Number of items.}
#'     \item{`key`}{Answer key.}
#'   }
#'
#' @examples
#' data(mcq_example)
#' result <- mcq_analysis(mcq_example$responses, mcq_example$key)
#' result
#'
#' @export
mcq_analysis <- function(
  responses,
  key,
  options = NULL,
  min_proportion = 0.05
) {
  inputs <- validate_inputs(responses, key)
  scored <- score_responses(inputs$responses, inputs$key)
  total <- rowSums(scored)
  difficulty <- item_difficulty(inputs$responses, inputs$key)
  pb <- point_biserial(inputs$responses, inputs$key, corrected = TRUE)
  D <- item_discrimination(
    inputs$responses,
    inputs$key,
    method = "discrimination_index"
  )
  da <- distractor_analysis(inputs$responses, inputs$key, options = options)
  de <- distractor_efficiency(
    inputs$responses,
    inputs$key,
    options = options,
    min_proportion = min_proportion
  )
  items_df <- data.frame(
    item = colnames(inputs$responses),
    key = inputs$key,
    difficulty = difficulty,
    point_biserial = pb,
    discrimination_index = D,
    distractor_efficiency = de,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
  out <- list(
    items = items_df,
    distractors = da,
    total_scores = total,
    n_students = nrow(scored),
    n_items = ncol(scored),
    key = inputs$key
  )
  class(out) <- "mcq_analysis"
  out
}

#' @export
print.mcq_analysis <- function(x, digits = 3, ...) {
  cat("Multiple-Choice Item Analysis\n")
  cat(strrep("-", 30), "\n", sep = "")
  cat("Students:", x$n_students, "\n")
  cat("Items:   ", x$n_items, "\n")
  cat(
    "Mean total score:",
    round(mean(x$total_scores, na.rm = TRUE), digits),
    " (SD =",
    round(stats::sd(x$total_scores, na.rm = TRUE), digits),
    ")\n"
  )
  cat("\nItem-level statistics:\n")
  display <- x$items
  display$difficulty <- round(display$difficulty, digits)
  display$point_biserial <- round(display$point_biserial, digits)
  display$discrimination_index <- round(display$discrimination_index, digits)
  print(display, row.names = FALSE)
  invisible(x)
}

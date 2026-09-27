#' Distractor analysis
#'
#' For each item, summarizes the selection frequency, proportion, and
#' point-biserial correlation with the total test score for every response
#' option (the key and all distractors). Distractor analysis is a core
#' classical test theory diagnostic for evaluating multiple-choice items:
#' the key should be the most-selected option and should have a positive
#' point-biserial correlation with total score, while each distractor
#' should be selected by at least some examinees and should have a
#' negative point-biserial correlation with total score (Haladyna, 2004).
#'
#' @param responses A matrix or data frame of student responses, with
#'   students in rows and items in columns.
#' @param key A vector of correct answers with length equal to the number
#'   of items.
#' @param options Optional character vector listing all possible response
#'   options (e.g., `c("A", "B", "C", "D")`). If `NULL` (default), the
#'   set of options is inferred from the unique values present in
#'   `responses`.
#'
#' @return A data frame in long format with one row per item-option
#'   combination, containing:
#'   \itemize{
#'     \item `item`: item identifier
#'     \item `option`: response option
#'     \item `is_key`: logical, `TRUE` if this option is the correct answer
#'     \item `frequency`: number of students selecting this option
#'     \item `proportion`: proportion of students selecting this option
#'     \item `point_biserial`: correlation between selecting this option
#'           and the total test score (using all items)
#'   }
#'
#' @references
#' Haladyna, T. M. (2004). *Developing and validating multiple-choice test
#' items* (3rd ed.). Lawrence Erlbaum Associates.
#'
#' @examples
#' set.seed(1)
#' responses <- matrix(
#'   sample(c("A", "B", "C", "D"), 200, replace = TRUE),
#'   nrow = 50, ncol = 4,
#'   dimnames = list(NULL, paste0("Q", 1:4))
#' )
#' key <- c("A", "B", "C", "A")
#' distractor_analysis(responses, key)
#'
#' @export
distractor_analysis <- function(responses, key, options = NULL) {
  inputs <- validate_inputs(responses, key)
  resp <- inputs$responses
  key <- inputs$key
  scored <- score_responses(resp, key)
  total <- rowSums(scored)
  if (is.null(options)) {
    options <- sort(unique(as.vector(resp)))
    options <- options[!is.na(options)]
  } else {
    options <- as.character(options)
  }
  out <- vector("list", ncol(resp))
  for (j in seq_len(ncol(resp))) {
    item_resp <- resp[, j]
    rows <- vector("list", length(options))
    for (k in seq_along(options)) {
      opt <- options[k]
      selected <- as.integer(item_resp == opt)
      freq <- sum(selected, na.rm = TRUE)
      prop <- mean(selected, na.rm = TRUE)
      if (
        stats::sd(selected, na.rm = TRUE) == 0 ||
          stats::sd(total) == 0
      ) {
        rpb <- NA_real_
      } else {
        rpb <- stats::cor(selected, total, use = "complete.obs")
      }
      rows[[k]] <- data.frame(
        item = colnames(resp)[j],
        option = opt,
        is_key = (opt == key[j]),
        frequency = freq,
        proportion = prop,
        point_biserial = rpb,
        stringsAsFactors = FALSE
      )
    }
    out[[j]] <- do.call(rbind, rows)
  }
  do.call(rbind, out)
}

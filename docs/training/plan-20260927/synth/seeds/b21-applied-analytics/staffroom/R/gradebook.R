#' Apply a late-submission penalty
#'
#' Reduces the percentage score of late work according to the course policy.
#'
#' @param pct Numeric vector of percentage scores (0-100).
#' @param days_late Numeric vector of days late; zero or `NA` means on time.
#' @param policy Late policy: a fixed deduction per day, a one-off cap, or no
#'   penalty.
#' @param rate Points deducted per day for `"per_day"`, or the cap for
#'   `"cap"`.
#' @return A numeric vector of penalised scores, never below zero.
#' @export
apply_late_penalty <- function(pct, days_late, policy = c("per_day", "cap", "none"),
                               rate = 10) {
  policy <- match.arg(policy)
  days_late[is.na(days_late)] <- 0
  late <- days_late > 0
  penalised <- switch(policy,
    per_day = pct - rate * ceiling(days_late),
    cap = ifelse(late, pmin(pct, rate), pct),
    none = pct
  )
  pmax(penalised, 0)
}

#' Drop the lowest scores within each category
#'
#' Marks the `n` lowest assignment scores per student and category as dropped,
#' as many syllabi allow for quizzes.
#'
#' @param scores A data frame of scores with columns `student_id`,
#'   `category`, `assignment` and `pct`.
#' @param n Number of scores to drop per student and category.
#' @param categories Categories the rule applies to; `NULL` means all.
#' @return `scores` with a logical `dropped` column.
#' @export
drop_lowest <- function(scores, n = 1L, categories = NULL) {
  if (is.null(categories)) {
    categories <- unique(scores$category)
  }
  scores %>%
    dplyr::group_by(student_id, category) %>%
    dplyr::mutate(
      rank_low = rank(pct, ties.method = "first", na.last = FALSE),
      dropped = category %in% categories & rank_low <= n & dplyr::n() > n
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-rank_low)
}

#' Weighted course percentage per student
#'
#' Averages the kept scores within each category, then combines categories
#' using the syllabus weights, renormalising when a category is empty.
#'
#' @param scores A data frame of scores with columns `student_id`,
#'   `category`, `assignment` and `pct`.
#' @param weights A named numeric vector of category weights.
#' @return A data frame with one row per student and columns `student_id`,
#'   `course_pct` and `weight_used`.
#' @export
course_percentages <- function(scores, weights) {
  unknown <- setdiff(unique(scores$category), names(weights))
  if (length(unknown) > 0) {
    stop("No weight for categories: ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  if (!"dropped" %in% names(scores)) {
    scores$dropped <- FALSE
  }
  scores %>%
    dplyr::filter(!dropped, !is.na(pct)) %>%
    dplyr::group_by(student_id, category) %>%
    dplyr::summarise(cat_pct = mean(pct), .groups = "drop") %>%
    dplyr::mutate(weight = unname(weights[category])) %>%
    dplyr::group_by(student_id) %>%
    dplyr::summarise(
      course_pct = sum(cat_pct * weight) / sum(weight),
      weight_used = sum(weight) / sum(weights),
      .groups = "drop"
    )
}

#' Map percentages to letter grades
#'
#' Uses the lower bounds in `scale`, highest first; anything below the lowest
#' bound receives `"F"`.
#'
#' @param pct Numeric vector of percentage scores (0-100).
#' @param scale A named numeric vector of lower bounds, named by grade.
#' @return A character vector of letter grades.
#' @family grade scales
#' @export
letter_grade <- function(pct, scale = c(A = 90, B = 80, C = 70, D = 60)) {
  scale <- sort(scale, decreasing = TRUE)
  out <- rep("F", length(pct))
  for (g in rev(names(scale))) {
    out[!is.na(pct) & pct >= scale[[g]]] <- g
  }
  out[is.na(pct)] <- NA_character_
  out
}

#' Map letter grades to grade points
#'
#' Looks up each letter in a points table; unknown letters give `NA`.
#'
#' @param grade Character vector of letter grades.
#' @param points A named numeric vector of grade points, named by grade.
#' @return A numeric vector of grade points.
#' @family grade scales
#' @export
grade_points <- function(grade, points = c(A = 4, B = 3, C = 2, D = 1, F = 0)) {
  out <- unname(points[grade])
  out[is.na(grade)] <- NA_real_
  out
}

#' Summarise a gradebook by student
#'
#' Counts submitted and missing work and the mean percentage for each
#' student.
#'
#' @param scores A data frame of scores with columns `student_id`,
#'   `category`, `assignment` and `pct`.
#' @return A data frame with one row per student.
#' @family gradebook summaries
#' @export
summarise_by_student <- function(scores) {
  scores %>%
    dplyr::group_by(student_id) %>%
    dplyr::summarise(
      n_assigned = dplyr::n(),
      n_missing = sum(is.na(pct)),
      mean_pct = mean(pct, na.rm = TRUE),
      min_pct = suppressWarnings(min(pct, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(n_missing), mean_pct)
}

#' Summarise a gradebook by assignment
#'
#' Counts submitted and missing work and the mean percentage for each
#' assignment.
#'
#' @param scores A data frame of scores with columns `student_id`,
#'   `category`, `assignment` and `pct`.
#' @return A data frame with one row per assignment.
#' @family gradebook summaries
#' @export
summarise_by_assignment <- function(scores) {
  scores %>%
    dplyr::group_by(category, assignment) %>%
    dplyr::summarise(
      n_assigned = dplyr::n(),
      n_missing = sum(is.na(pct)),
      mean_pct = mean(pct, na.rm = TRUE),
      min_pct = suppressWarnings(min(pct, na.rm = TRUE)),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(n_missing), mean_pct)
}

#' Produce final course grades
#'
#' Runs the full pipeline: late penalties, dropped scores, weighted
#' percentages, letters and grade points, with an incomplete flag for
#' students missing too much of the weight.
#'
#' @param scores A data frame of scores with columns `student_id`,
#'   `category`, `assignment` and `pct`.
#' @param weights A named numeric vector of category weights.
#' @param drop_n Number of scores to drop per student in `drop_categories`.
#' @param drop_categories Categories in which low scores may be dropped.
#' @param min_weight Minimum share of the total weight a student must have
#'   attempted to receive a grade.
#' @return A data frame with one row per student and columns `student_id`,
#'   `course_pct`, `letter`, `points` and `incomplete`.
#' @export
final_grades <- function(scores, weights, drop_n = 1L, drop_categories = "quiz",
                         min_weight = 0.75) {
  if ("days_late" %in% names(scores)) {
    scores <- scores %>%
      dplyr::mutate(pct = apply_late_penalty(pct, days_late))
  }
  pct_table <- scores %>%
    drop_lowest(n = drop_n, categories = drop_categories) %>%
    course_percentages(weights)
  pct_table %>%
    dplyr::mutate(
      incomplete = weight_used < min_weight,
      letter = dplyr::if_else(incomplete, "I", letter_grade(course_pct)),
      points = dplyr::if_else(incomplete, NA_real_, grade_points(letter))
    ) %>%
    dplyr::select(student_id, course_pct, letter, points, incomplete)
}

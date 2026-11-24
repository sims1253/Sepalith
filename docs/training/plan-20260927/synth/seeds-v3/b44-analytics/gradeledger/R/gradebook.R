#' Validate a grade book
#'
#' Checks the `student` column and each assessment component in turn.
#'
#' @param book A data frame with columns `student`, `homework`, `quizzes`,
#'   `midterm`, `final` and `project`, raw points per student.
#' @param maxPoints Named maximum points for each assessment component.
#' @return `book`, invisibly, when it is valid.
#' @export
validateGradebook = function(book, maxPoints) {
  if (!is.data.frame(book)) {
    stop("`book` must be a data frame.", call. = FALSE)
  }
  if (anyNA(book$student) || anyDuplicated(book$student) > 0L) {
    stop("Column `student` must hold unique, non-missing ids.", call. = FALSE)
  }

  # Validate `homework`
  x = book$homework
  limit = maxPoints[["homework"]]
  if (!is.numeric(x)) {
    stop(sprintf("Column `%s` must be numeric.", "homework"), call. = FALSE)
  }
  if (any(x < 0 | x > limit, na.rm = TRUE)) {
    stop(sprintf("Column `%s` must lie between 0 and %s.", "homework", limit), call. = FALSE)
  }
  # Validate `quizzes`
  x = book$quizzes
  limit = maxPoints[["quizzes"]]
  if (!is.numeric(x)) {
    stop(sprintf("Column `%s` must be numeric.", "quizzes"), call. = FALSE)
  }
  if (any(x < 0 | x > limit, na.rm = TRUE)) {
    stop(sprintf("Column `%s` must lie between 0 and %s.", "quizzes", limit), call. = FALSE)
  }
  # Validate `midterm`
  x = book$midterm
  limit = maxPoints[["midterm"]]
  if (!is.numeric(x)) {
    stop(sprintf("Column `%s` must be numeric.", "midterm"), call. = FALSE)
  }
  if (any(x < 0 | x > limit, na.rm = TRUE)) {
    stop(sprintf("Column `%s` must lie between 0 and %s.", "midterm", limit), call. = FALSE)
  }
  # Validate `final`
  x = book$final
  limit = maxPoints[["final"]]
  if (!is.numeric(x)) {
    stop(sprintf("Column `%s` must be numeric.", "final"), call. = FALSE)
  }
  if (any(x < 0 | x > limit, na.rm = TRUE)) {
    stop(sprintf("Column `%s` must lie between 0 and %s.", "final", limit), call. = FALSE)
  }
  # Validate `project`
  x = book$project
  limit = maxPoints[["project"]]
  if (!is.numeric(x)) {
    stop(sprintf("Column `%s` must be numeric.", "project"), call. = FALSE)
  }
  if (any(x < 0 | x > limit, na.rm = TRUE)) {
    stop(sprintf("Column `%s` must lie between 0 and %s.", "project", limit), call. = FALSE)
  }

  return(invisible(book))
}

#' Convert raw points to percentages
#'
#' One block per assessment component; values above the maximum are capped at
#' 100 and flagged.
#'
#' @inheritParams validateGradebook
#' @param digits Number of decimal places to keep.
#' @return A data frame with columns `student`, `homework`, `quizzes`,
#'   `midterm`, `final`, `project` (percentages) and `overMax`.
#' @export
componentPercent = function(book, maxPoints, digits = 1) {
  out = data.frame(student = book$student)
  overMax = logical(nrow(book))

  # Percent for `homework`
  raw = book$homework
  pct = 100 * raw / maxPoints[["homework"]]
  out$homework = round(pmin(pct, 100), digits)
  overMax = overMax | (!is.na(raw) & raw > maxPoints[["homework"]])
  # Percent for `quizzes`
  raw = book$quizzes
  pct = 100 * raw / maxPoints[["quizzes"]]
  out$quizzes = round(pmin(pct, 100), digits)
  overMax = overMax | (!is.na(raw) & raw > maxPoints[["quizzes"]])
  # Percent for `midterm`
  raw = book$midterm
  pct = 100 * raw / maxPoints[["midterm"]]
  out$midterm = round(pmin(pct, 100), digits)
  overMax = overMax | (!is.na(raw) & raw > maxPoints[["midterm"]])
  # Percent for `final`
  raw = book$final
  pct = 100 * raw / maxPoints[["final"]]
  out$final = round(pmin(pct, 100), digits)
  overMax = overMax | (!is.na(raw) & raw > maxPoints[["final"]])
  # Percent for `project`
  raw = book$project
  pct = 100 * raw / maxPoints[["project"]]
  out$project = round(pmin(pct, 100), digits)
  overMax = overMax | (!is.na(raw) & raw > maxPoints[["project"]])

  out$overMax = overMax
  return(out)
}

#' Weighted final grade
#'
#' Adds one weighted block per assessment component. Missing components count
#' as zero, or are dropped and the remaining weights rescaled.
#'
#' @param homework,quizzes,midterm,final,project Component percentages, 0-100.
#' @param weights Named weights for each assessment component.
#' @param missingAs Either `"zero"` or `"drop"`.
#' @return A numeric vector of final percentages, rounded to two decimals.
#' @export
finalGrade = function(homework, quizzes, midterm, final, project,
                      weights = c(homework = 0.2, quizzes = 0.1, midterm = 0.25,
                                  final = 0.35, project = 0.1),
                      missingAs = c("zero", "drop")) {
  missingAs = match.arg(missingAs)
  total = numeric(length(final))
  used = numeric(length(final))

  # Component `homework`
  if (any(homework < 0 | homework > 100, na.rm = TRUE)) {
    stop(sprintf("`%s` must lie between 0 and 100.", "homework"), call. = FALSE)
  }
  have = !is.na(homework)
  total = total + weights[["homework"]] * ifelse(have, homework, 0)
  used = used + weights[["homework"]] * (have | missingAs == "zero")
  # Component `quizzes`
  if (any(quizzes < 0 | quizzes > 100, na.rm = TRUE)) {
    stop(sprintf("`%s` must lie between 0 and 100.", "quizzes"), call. = FALSE)
  }
  have = !is.na(quizzes)
  total = total + weights[["quizzes"]] * ifelse(have, quizzes, 0)
  used = used + weights[["quizzes"]] * (have | missingAs == "zero")
  # Component `midterm`
  if (any(midterm < 0 | midterm > 100, na.rm = TRUE)) {
    stop(sprintf("`%s` must lie between 0 and 100.", "midterm"), call. = FALSE)
  }
  have = !is.na(midterm)
  total = total + weights[["midterm"]] * ifelse(have, midterm, 0)
  used = used + weights[["midterm"]] * (have | missingAs == "zero")
  # Component `final`
  if (any(final < 0 | final > 100, na.rm = TRUE)) {
    stop(sprintf("`%s` must lie between 0 and 100.", "final"), call. = FALSE)
  }
  have = !is.na(final)
  total = total + weights[["final"]] * ifelse(have, final, 0)
  used = used + weights[["final"]] * (have | missingAs == "zero")
  # Component `project`
  if (any(project < 0 | project > 100, na.rm = TRUE)) {
    stop(sprintf("`%s` must lie between 0 and 100.", "project"), call. = FALSE)
  }
  have = !is.na(project)
  total = total + weights[["project"]] * ifelse(have, project, 0)
  used = used + weights[["project"]] * (have | missingAs == "zero")

  score = ifelse(used > 0, total / used, NA_real_)
  return(round(score, 2))
}

#' Letter grades from percentages
#'
#' @param score A numeric vector of percentages.
#' @param scale One of `"standard"`, `"plusminus"`, `"passfail"` or `"honors"`.
#' @return A data frame with columns `score`, `letter` and `points`.
#' @export
letterGrade = function(score, scale = c("standard", "plusminus", "passfail", "honors")) {
  scale = match.arg(scale)
  if (!is.numeric(score)) {
    stop("`score` must be numeric.", call. = FALSE)
  }
  switch(scale,
    standard = {
      cuts = c(0, 60, 70, 80, 90)
      labels = c("F", "D", "C", "B", "A")
      points = c(0, 1, 2, 3, 4)
    },
    plusminus = {
      cuts = c(0, 60, 63, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97)
      labels = c("F", "D-", "D", "D+", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+")
      points = c(0, 0.7, 1, 1.3, 1.7, 2, 2.3, 2.7, 3, 3.3, 3.7, 4, 4)
    },
    passfail = {
      cuts = c(0, 60)
      labels = c("Fail", "Pass")
      points = c(NA_real_, NA_real_)
    },
    honors = {
      cuts = c(0, 60, 80, 90)
      labels = c("Fail", "Pass", "Merit", "Distinction")
      points = c(0, 2, 3, 4)
    }
  )

  idx = findInterval(score, cuts)
  idx[idx == 0L] = NA_integer_
  return(data.frame(score = score, letter = labels[idx], points = points[idx]))
}

#' Curve a vector of scores
#'
#' @param scores A numeric vector of percentages.
#' @param method One of `"none"`, `"flat"`, `"linear"`, `"sqrt"` or `"zscore"`.
#' @param target Mean score aimed for by `"flat"` and `"zscore"`.
#' @param sdTarget Standard deviation aimed for by `"zscore"`.
#' @param cap Highest score allowed after curving.
#' @return A list with elements `scores`, `method` and `shift`.
#' @export
curveScores = function(scores, method = c("none", "flat", "linear", "sqrt", "zscore"),
                       target = 75, sdTarget = 10, cap = 100) {
  method = match.arg(method)
  ok = !is.na(scores)
  avg = mean(scores[ok])
  top = max(scores[ok])
  spread = stats::sd(scores[ok])
  switch(method,
    none = {
      # Leave scores unchanged
      curved = scores
    },
    flat = {
      # Shift every score so the mean hits `target`
      curved = scores + (target - avg)
    },
    linear = {
      # Stretch so the top score reaches `cap`
      curved = scores * cap / top
    },
    sqrt = {
      # Square root curve on the 0 to `cap` scale
      curved = sqrt(scores / cap) * cap
    },
    zscore = {
      # Rescale to mean `target` and spread `sdTarget`
      curved = target + sdTarget * (scores - avg) / spread
    }
  )
  curved = pmin(pmax(curved, 0), cap)
  return(list(scores = round(curved, 1), method = method, shift = mean(curved[ok]) - avg))
}

#' Summarise each assessment component
#'
#' @inheritParams validateGradebook
#' @return A matrix with one row per component (`homework`, `quizzes`,
#'   `midterm`, `final`, `project`) and columns `n`, `missing`, `mean`,
#'   `median`, `min` and `max`.
#' @export
componentSummary = function(book) {
  out = list()

  # Summary of `homework`
  x = book$homework
  ok = x[!is.na(x)]
  out$homework = c(
    n = length(ok), missing = sum(is.na(x)),
    mean = mean(ok), median = stats::median(ok), min = min(ok), max = max(ok)
  )
  # Summary of `quizzes`
  x = book$quizzes
  ok = x[!is.na(x)]
  out$quizzes = c(
    n = length(ok), missing = sum(is.na(x)),
    mean = mean(ok), median = stats::median(ok), min = min(ok), max = max(ok)
  )
  # Summary of `midterm`
  x = book$midterm
  ok = x[!is.na(x)]
  out$midterm = c(
    n = length(ok), missing = sum(is.na(x)),
    mean = mean(ok), median = stats::median(ok), min = min(ok), max = max(ok)
  )
  # Summary of `final`
  x = book$final
  ok = x[!is.na(x)]
  out$final = c(
    n = length(ok), missing = sum(is.na(x)),
    mean = mean(ok), median = stats::median(ok), min = min(ok), max = max(ok)
  )
  # Summary of `project`
  x = book$project
  ok = x[!is.na(x)]
  out$project = c(
    n = length(ok), missing = sum(is.na(x)),
    mean = mean(ok), median = stats::median(ok), min = min(ok), max = max(ok)
  )

  return(do.call(rbind, out))
}

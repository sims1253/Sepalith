#' @title Score a rubric
#' @description Converts the level on each criterion to weighted points out of 100.
#' @param clarity,evidence,structure,mechanics,originality Integer levels per
#'   criterion, from 1 to `levels`.
#' @param weights Named weights for each criterion, summing to one.
#' @param levels Number of levels on the rubric.
#' @return A data frame with one points column per criterion and `total`.
#' @export
scoreRubric = function(clarity, evidence, structure, mechanics, originality,
                       weights = c(clarity = 0.2, evidence = 0.3, structure = 0.2,
                                   mechanics = 0.1, originality = 0.2),
                       levels = 4L) {
  out = list()
  total = numeric(length(clarity))

  # Criterion `clarity`
  if (!all(clarity %in% seq_len(levels))) {
    stop(sprintf("`%s` must hold levels 1 to %d.", "clarity", levels), call. = FALSE)
  }
  pts = 100 * weights[["clarity"]] * clarity / levels
  out$clarity = pts
  total = total + pts
  # Criterion `evidence`
  if (!all(evidence %in% seq_len(levels))) {
    stop(sprintf("`%s` must hold levels 1 to %d.", "evidence", levels), call. = FALSE)
  }
  pts = 100 * weights[["evidence"]] * evidence / levels
  out$evidence = pts
  total = total + pts
  # Criterion `structure`
  if (!all(structure %in% seq_len(levels))) {
    stop(sprintf("`%s` must hold levels 1 to %d.", "structure", levels), call. = FALSE)
  }
  pts = 100 * weights[["structure"]] * structure / levels
  out$structure = pts
  total = total + pts
  # Criterion `mechanics`
  if (!all(mechanics %in% seq_len(levels))) {
    stop(sprintf("`%s` must hold levels 1 to %d.", "mechanics", levels), call. = FALSE)
  }
  pts = 100 * weights[["mechanics"]] * mechanics / levels
  out$mechanics = pts
  total = total + pts
  # Criterion `originality`
  if (!all(originality %in% seq_len(levels))) {
    stop(sprintf("`%s` must hold levels 1 to %d.", "originality", levels), call. = FALSE)
  }
  pts = 100 * weights[["originality"]] * originality / levels
  out$originality = pts
  total = total + pts

  out$total = total
  return(as.data.frame(out))
}

#' @title Written feedback per criterion
#' @description Turns rubric levels into a label per criterion and names the
#'   weakest criterion as the next focus.
#' @param scores A data frame with columns `student`, `clarity`, `evidence`,
#'   `structure`, `mechanics` and `originality`, levels per criterion.
#' @param labels Level labels, lowest first.
#' @return A data frame with columns `student`, one text column per criterion
#'   and `focus`.
#' @export
rubricFeedback = function(scores,
                          labels = c("Beginning", "Developing", "Proficient", "Exemplary")) {
  n = nrow(scores)
  top = length(labels)
  weakest = rep(NA_character_, n)
  weakestLevel = rep(Inf, n)
  out = data.frame(student = scores$student)

  # Feedback on `clarity`
  lvl = scores$clarity
  out$clarity = sprintf("%s (%d/%d)", labels[lvl], lvl, top)
  lower = lvl < weakestLevel
  weakest[lower] = "clarity"
  weakestLevel[lower] = lvl[lower]
  # Feedback on `evidence`
  lvl = scores$evidence
  out$evidence = sprintf("%s (%d/%d)", labels[lvl], lvl, top)
  lower = lvl < weakestLevel
  weakest[lower] = "evidence"
  weakestLevel[lower] = lvl[lower]
  # Feedback on `structure`
  lvl = scores$structure
  out$structure = sprintf("%s (%d/%d)", labels[lvl], lvl, top)
  lower = lvl < weakestLevel
  weakest[lower] = "structure"
  weakestLevel[lower] = lvl[lower]
  # Feedback on `mechanics`
  lvl = scores$mechanics
  out$mechanics = sprintf("%s (%d/%d)", labels[lvl], lvl, top)
  lower = lvl < weakestLevel
  weakest[lower] = "mechanics"
  weakestLevel[lower] = lvl[lower]
  # Feedback on `originality`
  lvl = scores$originality
  out$originality = sprintf("%s (%d/%d)", labels[lvl], lvl, top)
  lower = lvl < weakestLevel
  weakest[lower] = "originality"
  weakestLevel[lower] = lvl[lower]

  out$focus = ifelse(weakestLevel >= top, "None", weakest)
  return(out)
}

#' @title Combine two raters
#' @description Merges the levels given by two raters into one score.
#' @param raterA,raterB Numeric levels from the first and second rater.
#' @param method One of `"mean"`, `"min"`, `"max"`, `"third"` or `"consensus"`.
#' @param raterC Levels from a third rater, used by `"third"`.
#' @param tolerance Largest gap between `raterA` and `raterB` that counts as
#'   agreement.
#' @return A data frame with columns `raterA`, `raterB`, `gap`, `combined` and
#'   `resolved`.
#' @export
combineRaters = function(raterA, raterB, method = c("mean", "min", "max", "third", "consensus"),
                         raterC = NULL, tolerance = 1) {
  method = match.arg(method)
  if (length(raterA) != length(raterB)) {
    stop("`raterA` and `raterB` must have the same length.", call. = FALSE)
  }
  gap = abs(raterA - raterB)
  switch(method,
    mean = {
      # Average of both raters
      combined = (raterA + raterB) / 2
      resolved = rep(TRUE, length(gap))
    },
    min = {
      # Lower of both raters
      combined = pmin(raterA, raterB)
      resolved = rep(TRUE, length(gap))
    },
    max = {
      # Higher of both raters
      combined = pmax(raterA, raterB)
      resolved = rep(TRUE, length(gap))
    },
    third = {
      # Average, with `raterC` settling gaps above `tolerance`
      if (is.null(raterC)) {
        stop("`raterC` is needed for method \"third\".", call. = FALSE)
      }
      combined = (raterA + raterB) / 2
      far = gap > tolerance
      combined[far] = raterC[far]
      resolved = rep(TRUE, length(gap))
    },
    consensus = {
      # Average, keeping only gaps within `tolerance`
      combined = (raterA + raterB) / 2
      resolved = gap <= tolerance
      combined[!resolved] = NA_real_
    }
  )
  return(data.frame(raterA = raterA, raterB = raterB, gap = gap, combined = combined,
                    resolved = resolved))
}

#' @title Agreement between two raters per criterion
#' @param raterA,raterB Data frames with columns `clarity`, `evidence`,
#'   `structure`, `mechanics` and `originality`, levels per criterion, rows in
#'   the same student order.
#' @return A data frame with columns `criterion`, `exact`, `adjacent` and
#'   `meanDiff`, one row per criterion.
#' @export
rubricAgreement = function(raterA, raterB) {
  if (nrow(raterA) != nrow(raterB)) {
    stop("`raterA` and `raterB` must have the same number of rows.", call. = FALSE)
  }
  criteria = c("clarity", "evidence", "structure", "mechanics", "originality")
  out = data.frame(criterion = criteria, exact = NA_real_, adjacent = NA_real_,
                   meanDiff = NA_real_)

  # Agreement on `clarity`
  d = raterA$clarity - raterB$clarity
  out$exact[1] = mean(d == 0)
  out$adjacent[1] = mean(abs(d) <= 1)
  out$meanDiff[1] = mean(d)
  # Agreement on `evidence`
  d = raterA$evidence - raterB$evidence
  out$exact[2] = mean(d == 0)
  out$adjacent[2] = mean(abs(d) <= 1)
  out$meanDiff[2] = mean(d)
  # Agreement on `structure`
  d = raterA$structure - raterB$structure
  out$exact[3] = mean(d == 0)
  out$adjacent[3] = mean(abs(d) <= 1)
  out$meanDiff[3] = mean(d)
  # Agreement on `mechanics`
  d = raterA$mechanics - raterB$mechanics
  out$exact[4] = mean(d == 0)
  out$adjacent[4] = mean(abs(d) <= 1)
  out$meanDiff[4] = mean(d)
  # Agreement on `originality`
  d = raterA$originality - raterB$originality
  out$exact[5] = mean(d == 0)
  out$adjacent[5] = mean(abs(d) <= 1)
  out$meanDiff[5] = mean(d)

  return(out)
}

#' @title Rescale rubric totals
#' @param total Rubric totals.
#' @param maxTotal Highest possible total.
#' @param to One of `"percent"`, `"points"`, `"gpa"` or `"band"`.
#' @param points Points available when `to` is `"points"`.
#' @return A data frame with columns `total`, `value` and `unit`.
#' @export
rescaleRubric = function(total, maxTotal = 100, to = c("percent", "points", "gpa", "band"),
                         points = 20) {
  to = match.arg(to)
  if (any(total < 0 | total > maxTotal, na.rm = TRUE)) {
    stop("`total` must lie between 0 and `maxTotal`.", call. = FALSE)
  }
  share = total / maxTotal
  switch(to,
    percent = {
      value = round(100 * share, 1)
      unit = "percent"
    },
    points = {
      value = round(points * share, 1)
      unit = "points"
    },
    gpa = {
      value = round(4 * share, 2)
      unit = "gpa"
    },
    band = {
      value = as.character(cut(share, c(0, 0.5, 0.65, 0.8, 1), include.lowest = TRUE,
                               labels = c("Below", "Approaching", "Meeting", "Exceeding")))
      unit = "band"
    }
  )
  return(data.frame(total = total, value = value, unit = unit))
}

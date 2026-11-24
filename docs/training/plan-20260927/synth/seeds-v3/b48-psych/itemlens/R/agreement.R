#' Agreement weights for ordered categories
#'
#' @description Builds a k-by-k weight matrix for weighted agreement.
#' @param k Number of rating categories.
#' @param type One of `"identity"`, `"linear"`, `"quadratic"` or `"adjacent"`.
#' @return A numeric k-by-k matrix with ones on the diagonal.
#' @keywords internal
kappaWeights = function(k, type = c("identity", "linear", "quadratic", "adjacent")) {
  type = match.arg(type)
  if (k < 2) {
    stop("`k` must be at least 2.", call. = FALSE)
  }
  dist = abs(outer(seq_len(k), seq_len(k), "-"))

  switch(type,
    identity = {
      (dist == 0) * 1
    },
    linear = {
      1 - dist / (k - 1)
    },
    quadratic = {
      1 - (dist / (k - 1))^2
    },
    adjacent = {
      (dist <= 1) * 1
    }
  )
}

#' Agreement between two raters
#'
#' @description Chance-corrected or raw agreement on categorical ratings.
#' @param r1,r2 Vectors of ratings by the first and second rater.
#' @param method One of `"percent"`, `"cohen"`, `"scott"`, `"weighted"` or
#'   `"gwet"`.
#' @param weights For `"weighted"`, a weight type passed to [kappaWeights()].
#' @return A single number with attribute `method`.
#' @export
agreementCoef = function(r1, r2, method = c("percent", "cohen", "scott", "weighted", "gwet"),
                         weights = "quadratic") {
  method = match.arg(method)
  keep = !is.na(r1) & !is.na(r2)
  levs = sort(unique(c(r1[keep], r2[keep])))
  tab = table(factor(r1[keep], levs), factor(r2[keep], levs))
  p = tab / sum(tab)
  k = length(levs)
  rowP = rowSums(p)
  colP = colSums(p)
  observed = sum(diag(p))

  value = switch(method,
    percent = {
      observed
    },
    cohen = {
      expected = sum(rowP * colP)
      (observed - expected) / (1 - expected)
    },
    scott = {
      pooled = (rowP + colP) / 2
      expected = sum(pooled^2)
      (observed - expected) / (1 - expected)
    },
    weighted = {
      w = kappaWeights(k, weights)
      observedW = sum(w * p)
      expectedW = sum(w * outer(rowP, colP))
      (observedW - expectedW) / (1 - expectedW)
    },
    gwet = {
      pooled = (rowP + colP) / 2
      expected = sum(pooled * (1 - pooled)) / (k - 1)
      (observed - expected) / (1 - expected)
    }
  )
  value = unname(value)
  attr(value, "method") = method
  value
}

#' Intraclass correlation
#'
#' @description ICC for a subjects-by-raters matrix under one model.
#' @param ratings A numeric matrix, one row per subject and one column per rater.
#' @param model One of `"oneway"`, `"agreement"` or `"consistency"`.
#' @param unit Either `"single"` or `"average"`.
#' @return A single number with attributes `model` and `unit`.
#' @export
iccEstimate = function(ratings, model = c("oneway", "agreement", "consistency"),
                       unit = c("single", "average")) {
  model = match.arg(model)
  unit = match.arg(unit)
  ratings = as.matrix(ratings)
  ratings = ratings[stats::complete.cases(ratings), , drop = FALSE]
  n = nrow(ratings)
  k = ncol(ratings)
  grand = mean(ratings)

  # Mean squares of the two-way layout
  ssRows = k * sum((rowMeans(ratings) - grand)^2)
  ssCols = n * sum((colMeans(ratings) - grand)^2)
  ssTotal = sum((ratings - grand)^2)
  msRows = ssRows / (n - 1)
  msCols = ssCols / (k - 1)
  msError = (ssTotal - ssRows - ssCols) / ((n - 1) * (k - 1))
  msWithin = (ssTotal - ssRows) / (n * (k - 1))

  value = switch(model,
    oneway = {
      if (unit == "single") {
        (msRows - msWithin) / (msRows + (k - 1) * msWithin)
      } else {
        (msRows - msWithin) / msRows
      }
    },
    agreement = {
      if (unit == "single") {
        (msRows - msError) / (msRows + (k - 1) * msError + k * (msCols - msError) / n)
      } else {
        (msRows - msError) / (msRows + (msCols - msError) / n)
      }
    },
    consistency = {
      if (unit == "single") {
        (msRows - msError) / (msRows + (k - 1) * msError)
      } else {
        (msRows - msError) / msRows
      }
    }
  )
  attr(value, "model") = model
  attr(value, "unit") = unit
  value
}

#' Summary of a multi-rater study
#'
#' @description Descriptive agreement statistics for many raters.
#' @param ratings A numeric matrix, one row per subject and one column per rater.
#' @return A list with elements `nSubjects`, `nRaters`, `raterMean`, `raterSd`,
#'   `exactAgreement` and `fleiss`.
#' @export
raterSummary = function(ratings) {
  ratings = as.matrix(ratings)
  ratings = ratings[stats::complete.cases(ratings), , drop = FALSE]
  if (ncol(ratings) < 2) {
    stop("`ratings` needs at least two raters.", call. = FALSE)
  }

  # 1. Subjects
  nSubjects = nrow(ratings)
  # 2. Raters
  nRaters = ncol(ratings)
  # 3. Mean rating per rater
  raterMean = colMeans(ratings)
  # 4. Standard deviation per rater
  raterSd = apply(ratings, 2, stats::sd)
  # 5. Share of subjects on which all raters agree
  exactAgreement = mean(apply(ratings, 1, function(row) length(unique(row)) == 1))
  # 6. Fleiss' kappa over all categories
  levs = sort(unique(c(ratings)))
  counts = t(apply(ratings, 1, function(row) table(factor(row, levs))))
  pj = colSums(counts) / (nSubjects * nRaters)
  pSubject = (rowSums(counts^2) - nRaters) / (nRaters * (nRaters - 1))
  fleiss = (mean(pSubject) - sum(pj^2)) / (1 - sum(pj^2))

  list(
    nSubjects = nSubjects,
    nRaters = nRaters,
    raterMean = raterMean,
    raterSd = raterSd,
    exactAgreement = exactAgreement,
    fleiss = fleiss
  )
}

#' Convergent and discriminant validity coefficients
#'
#' @description Correlates a scale with measures expected to relate or not.
#' @param scores A data frame of scale scores, one column per measure.
#' @param target Name of the scale under study.
#' @param convergent,discriminant Character vectors naming the measures
#'   expected to correlate highly and weakly with `target`.
#' @param threshold Absolute correlation separating high from low.
#' @return A data frame with columns `measure`, `type`, `r` and `supports`.
#' @export
validityCoefs = function(scores, target, convergent, discriminant, threshold = 0.3) {
  if (!target %in% names(scores)) {
    stop("`target` must be a column of `scores`.", call. = FALSE)
  }
  x = scores[[target]]

  # Convergent measures: support when |r| reaches the threshold
  if (!all(convergent %in% names(scores))) {
    stop(sprintf("Some `%s` measures are not columns of `scores`.", "convergent"),
      call. = FALSE)
  }
  rConv = vapply(convergent, function(m) {
    stats::cor(x, scores[[m]], use = "pairwise.complete.obs")
  }, numeric(1))
  conv = data.frame(measure = convergent, type = "convergent", r = unname(rConv))
  conv$supports = abs(conv$r) >= threshold

  # Discriminant measures: support when |r| stays below the threshold
  if (!all(discriminant %in% names(scores))) {
    stop(sprintf("Some `%s` measures are not columns of `scores`.", "discriminant"),
      call. = FALSE)
  }
  rDisc = vapply(discriminant, function(m) {
    stats::cor(x, scores[[m]], use = "pairwise.complete.obs")
  }, numeric(1))
  disc = data.frame(measure = discriminant, type = "discriminant", r = unname(rDisc))
  disc$supports = abs(disc$r) < threshold

  rbind(conv, disc)
}

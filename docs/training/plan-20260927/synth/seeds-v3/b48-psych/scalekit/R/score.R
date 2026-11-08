#' Score the wellbeing questionnaire
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param energy,mood,sleep,social Character vectors naming the items of each
#'   subscale.
#' @param method Either `"mean"` or `"sum"`.
#' @param min_items Smallest share of answered items needed for a score.
#' @return A data frame with columns `energy`, `mood`, `sleep`, `social` and
#'   `total`.
#' @export
score_wellbeing <- function(data, energy, mood, sleep, social,
                            method = c("mean", "sum"), min_items = 0.5) {
  method <- match.arg(method)
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  combine <- if (method == "mean") rowMeans else rowSums
  out <- data.frame(row.names = seq_len(nrow(data)))

  # Score `energy`
  absent <- setdiff(energy, names(data))
  if (length(absent) > 0) {
    stop(sprintf("Items of `%s` not found: %s", "energy", toString(absent)), call. = FALSE)
  }
  answered <- rowMeans(!is.na(data[energy]))
  out$energy <- combine(data[energy], na.rm = TRUE)
  out$energy[answered < min_items] <- NA_real_

  # Score `mood`
  absent <- setdiff(mood, names(data))
  if (length(absent) > 0) {
    stop(sprintf("Items of `%s` not found: %s", "mood", toString(absent)), call. = FALSE)
  }
  answered <- rowMeans(!is.na(data[mood]))
  out$mood <- combine(data[mood], na.rm = TRUE)
  out$mood[answered < min_items] <- NA_real_

  # Score `sleep`
  absent <- setdiff(sleep, names(data))
  if (length(absent) > 0) {
    stop(sprintf("Items of `%s` not found: %s", "sleep", toString(absent)), call. = FALSE)
  }
  answered <- rowMeans(!is.na(data[sleep]))
  out$sleep <- combine(data[sleep], na.rm = TRUE)
  out$sleep[answered < min_items] <- NA_real_

  # Score `social`
  absent <- setdiff(social, names(data))
  if (length(absent) > 0) {
    stop(sprintf("Items of `%s` not found: %s", "social", toString(absent)), call. = FALSE)
  }
  answered <- rowMeans(!is.na(data[social]))
  out$social <- combine(data[social], na.rm = TRUE)
  out$social[answered < min_items] <- NA_real_

  out$total <- combine(out[c("energy", "mood", "sleep", "social")])
  return(out)
}

#' Score one subscale
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items of the subscale.
#' @param method One of `"sum"`, `"mean"`, `"prorated"`, `"median"` or `"pomp"`.
#' @param min_value,max_value The lowest and highest response option.
#' @return A numeric vector with attributes `method` and `n_answered`.
#' @export
score_subscale <- function(data, items,
                           method = c("sum", "mean", "prorated", "median", "pomp"),
                           min_value = 1, max_value = 5) {
  method <- match.arg(method)
  if (!all(items %in% names(data))) {
    stop("All `items` must be columns of `data`.", call. = FALSE)
  }
  if (min_value >= max_value) {
    stop("`min_value` must be smaller than `max_value`.", call. = FALSE)
  }
  responses <- as.matrix(data[items])
  n_answered <- rowSums(!is.na(responses))
  n_items <- length(items)

  score <- switch(method,
    sum = {
      total <- rowSums(responses, na.rm = TRUE)
      total[n_answered < n_items] <- NA_real_
      total
    },
    mean = {
      avg <- rowMeans(responses, na.rm = TRUE)
      avg[n_answered == 0] <- NA_real_
      avg
    },
    prorated = {
      avg <- rowMeans(responses, na.rm = TRUE)
      avg[n_answered == 0] <- NA_real_
      avg * n_items
    },
    median = {
      mid <- apply(responses, 1, stats::median, na.rm = TRUE)
      mid[n_answered == 0] <- NA_real_
      mid
    },
    pomp = {
      avg <- rowMeans(responses, na.rm = TRUE)
      avg[n_answered == 0] <- NA_real_
      100 * (avg - min_value) / (max_value - min_value)
    }
  )
  score <- unname(score)
  attr(score, "method") <- method
  attr(score, "n_answered") <- unname(n_answered)
  return(score)
}

#' Score the burnout inventory
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param exhaustion,cynicism,efficacy Character vectors naming the items of
#'   each subscale.
#' @param cutoffs Named numeric vector of sum-score cutoffs per subscale.
#' @return A data frame with a score column and a `_flag` column per subscale,
#'   plus `n_flags`.
#' @export
score_burnout <- function(data, exhaustion, cynicism, efficacy,
                          cutoffs = c(exhaustion = 27, cynicism = 10, efficacy = 33)) {
  # Efficacy is flagged when low, the other subscales when high
  direction <- c(exhaustion = 1, cynicism = 1, efficacy = -1)
  if (!all(names(direction) %in% names(cutoffs))) {
    stop("`cutoffs` must be named `exhaustion`, `cynicism` and `efficacy`.", call. = FALSE)
  }
  out <- data.frame(row.names = seq_len(nrow(data)))

  # Exhaustion
  if (!all(exhaustion %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "exhaustion"), call. = FALSE)
  }
  out$exhaustion <- rowSums(data[exhaustion])
  out$exhaustion_flag <-
    direction[["exhaustion"]] * (out$exhaustion - cutoffs[["exhaustion"]]) >= 0

  # Cynicism
  if (!all(cynicism %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "cynicism"), call. = FALSE)
  }
  out$cynicism <- rowSums(data[cynicism])
  out$cynicism_flag <-
    direction[["cynicism"]] * (out$cynicism - cutoffs[["cynicism"]]) >= 0

  # Efficacy
  if (!all(efficacy %in% names(data))) {
    stop(sprintf("Some `%s` items are not columns of `data`.", "efficacy"), call. = FALSE)
  }
  out$efficacy <- rowSums(data[efficacy])
  out$efficacy_flag <-
    direction[["efficacy"]] * (out$efficacy - cutoffs[["efficacy"]]) >= 0

  out$n_flags <- rowSums(out[c("exhaustion_flag", "cynicism_flag", "efficacy_flag")])
  return(out)
}

#' Classify depression, anxiety and stress scores into severity bands
#'
#' @param scores A data frame with columns `depression`, `anxiety` and `stress`.
#' @param depression,anxiety,stress Increasing numeric vectors of four lower
#'   bounds for the `mild`, `moderate`, `severe` and `extreme` bands.
#' @return A data frame of ordered factors with the same columns as `scores`.
#' @export
classify_severity <- function(scores, depression = c(5, 7, 11, 14),
                              anxiety = c(4, 6, 8, 10), stress = c(8, 10, 13, 17)) {
  bands <- c("normal", "mild", "moderate", "severe", "extreme")
  out <- data.frame(row.names = seq_len(nrow(scores)))

  # Depression
  if (length(depression) != 4L || is.unsorted(depression, strictly = TRUE)) {
    stop(sprintf("`%s` must be four increasing bounds.", "depression"), call. = FALSE)
  }
  out$depression <- cut(scores$depression, c(-Inf, depression, Inf),
    labels = bands, right = FALSE, ordered_result = TRUE
  )

  # Anxiety
  if (length(anxiety) != 4L || is.unsorted(anxiety, strictly = TRUE)) {
    stop(sprintf("`%s` must be four increasing bounds.", "anxiety"), call. = FALSE)
  }
  out$anxiety <- cut(scores$anxiety, c(-Inf, anxiety, Inf),
    labels = bands, right = FALSE, ordered_result = TRUE
  )

  # Stress
  if (length(stress) != 4L || is.unsorted(stress, strictly = TRUE)) {
    stop(sprintf("`%s` must be four increasing bounds.", "stress"), call. = FALSE)
  }
  out$stress <- cut(scores$stress, c(-Inf, stress, Inf),
    labels = bands, right = FALSE, ordered_result = TRUE
  )

  return(out)
}

#' Summarise scale scores
#'
#' @param scores A data frame of scale scores, one column per subscale.
#' @param subscales Character vector of columns to summarise.
#' @return A data frame with one row per subscale and columns `subscale`, `n`,
#'   `missing`, `mean`, `sd`, `min`, `max` and `status`.
#' @keywords internal
summarise_scores <- function(scores, subscales = names(scores)) {
  stats <- c("n", "missing", "mean", "sd", "min", "max")
  if (!all(subscales %in% names(scores))) {
    stop("All `subscales` must be columns of `scores`.", call. = FALSE)
  }
  out <- matrix(NA_real_, nrow = length(subscales), ncol = length(stats),
    dimnames = list(subscales, stats)
  )

  for (s in subscales) {
    x <- scores[[s]]
    ok <- x[!is.na(x)]
    out[s, "n"] <- length(ok)
    out[s, "missing"] <- sum(is.na(x))
    if (length(ok) == 0) {
      next
    }
    out[s, "mean"] <- mean(ok)
    out[s, "sd"] <- if (length(ok) > 1) stats::sd(ok) else NA_real_
    out[s, "min"] <- min(ok)
    out[s, "max"] <- max(ok)
  }

  result <- data.frame(subscale = subscales, out, row.names = NULL)
  result$status <- ifelse(result$missing > 0, "partial", "complete")
  return(result)
}

#' Validate a scoring specification
#'
#' @param spec A list with elements `items`, `min_value`, `max_value`,
#'   `min_items`, `method` and `reverse`.
#' @return `spec`, invisibly, with defaults filled in for `min_items`,
#'   `method` and `reverse`.
#' @keywords internal
validate_scoring_spec <- function(spec) {
  fields <- c("items", "min_value", "max_value", "min_items", "method", "reverse")
  unknown <- setdiff(names(spec), fields)
  if (length(unknown) > 0) {
    stop(sprintf("Unknown spec fields: %s", toString(unknown)), call. = FALSE)
  }

  # Validate `items`
  if (!is.character(spec$items) || length(spec$items) == 0) {
    stop("`items` must be a non-empty character vector.", call. = FALSE)
  }
  # Validate `min_value`
  if (!is.numeric(spec$min_value) || length(spec$min_value) != 1L) {
    stop("`min_value` must be a single number.", call. = FALSE)
  }
  # Validate `max_value`
  if (!is.numeric(spec$max_value) || length(spec$max_value) != 1L) {
    stop("`max_value` must be a single number.", call. = FALSE)
  }
  if (spec$max_value <= spec$min_value) {
    stop("`max_value` must be larger than `min_value`.", call. = FALSE)
  }
  # Validate `min_items`
  if (is.null(spec$min_items)) {
    spec$min_items <- 0.5
  }
  if (!is.numeric(spec$min_items) || spec$min_items <= 0 || spec$min_items > 1) {
    stop("`min_items` must be a share in (0, 1].", call. = FALSE)
  }
  # Validate `method`
  if (is.null(spec$method)) {
    spec$method <- "mean"
  }
  if (!spec$method %in% c("sum", "mean", "prorated", "median", "pomp")) {
    stop("`method` must be one of the scoring methods.", call. = FALSE)
  }
  # Validate `reverse`
  if (is.null(spec$reverse)) {
    spec$reverse <- character()
  }
  if (!all(spec$reverse %in% spec$items)) {
    stop("`reverse` must be a subset of `items`.", call. = FALSE)
  }

  invisible(spec)
}

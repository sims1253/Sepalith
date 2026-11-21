#' One item statistic for every item
#'
#' @description Computes the chosen statistic for each column of `items`.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param stat One of `"difficulty"`, `"variance"`, `"itemTotal"`,
#'   `"itemRest"`, `"alphaIfDeleted"` or `"discrimination"`.
#' @param maxScore The highest possible item score, used by `"difficulty"`.
#' @return A named numeric vector, one value per item.
#' @export
itemStatistic = function(items, stat = c("difficulty", "variance", "itemTotal", "itemRest",
                                         "alphaIfDeleted", "discrimination"),
                         maxScore = 1) {
  stat = match.arg(stat)
  items = as.matrix(items)
  items = items[stats::complete.cases(items), , drop = FALSE]
  k = ncol(items)
  total = rowSums(items)

  values = switch(stat,
    difficulty = {
      colMeans(items) / maxScore
    },
    variance = {
      apply(items, 2, stats::var)
    },
    itemTotal = {
      apply(items, 2, function(col) stats::cor(col, total))
    },
    itemRest = {
      vapply(seq_len(k), function(j) {
        stats::cor(items[, j], total - items[, j])
      }, numeric(1))
    },
    alphaIfDeleted = {
      vapply(seq_len(k), function(j) {
        rest = items[, -j, drop = FALSE]
        covMat = stats::cov(rest)
        ((k - 1) / (k - 2)) * (1 - sum(diag(covMat)) / sum(covMat))
      }, numeric(1))
    },
    discrimination = {
      # Upper minus lower 27 percent on the total score
      cut = stats::quantile(total, c(0.27, 0.73))
      upper = items[total >= cut[2], , drop = FALSE]
      lower = items[total <= cut[1], , drop = FALSE]
      (colMeans(upper) - colMeans(lower)) / maxScore
    }
  )
  names(values) = colnames(items)
  values
}

#' Classical item analysis table
#'
#' @description Collects the usual item statistics in one data frame.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param maxScore The highest possible item score.
#' @return A data frame with columns `item`, `mean`, `sd`, `difficulty`,
#'   `itemRest` and `alphaIfDeleted`.
#' @export
itemAnalysis = function(items, maxScore = 1) {
  items = as.matrix(items)
  if (is.null(colnames(items))) {
    colnames(items) = paste0("item", seq_len(ncol(items)))
  }
  if (ncol(items) < 3) {
    stop("`items` must have at least three columns.", call. = FALSE)
  }
  out = data.frame(item = colnames(items))

  # Column `mean`
  out$mean = unname(colMeans(items, na.rm = TRUE))

  # Column `sd`
  out$sd = unname(apply(items, 2, stats::sd, na.rm = TRUE))

  # Column `difficulty`
  out$difficulty = unname(itemStatistic(items, "difficulty", maxScore = maxScore))

  # Column `itemRest`
  out$itemRest = unname(itemStatistic(items, "itemRest"))

  # Column `alphaIfDeleted`
  out$alphaIfDeleted = unname(itemStatistic(items, "alphaIfDeleted"))

  out
}

#' Flag items that miss quality criteria
#'
#' @description Checks an item analysis table against lower and upper limits.
#' @param stats A data frame as returned by [itemAnalysis()].
#' @param minDifficulty,maxDifficulty Acceptable range of `difficulty`.
#' @param minItemRest Smallest acceptable `itemRest` correlation.
#' @param alpha The alpha of the full scale, compared with `alphaIfDeleted`.
#' @return `stats` with logical columns `tooHard`, `tooEasy`, `weak`,
#'   `hurtsAlpha` and `flagged`.
#' @export
flagItems = function(stats, minDifficulty = 0.2, maxDifficulty = 0.8, minItemRest = 0.3,
                     alpha = NULL) {
  needed = c("difficulty", "itemRest", "alphaIfDeleted")
  if (!all(needed %in% names(stats))) {
    stop("`stats` must have columns `difficulty`, `itemRest` and `alphaIfDeleted`.",
      call. = FALSE)
  }

  # Lower limit on difficulty
  stats$tooHard = stats$difficulty < minDifficulty

  # Upper limit on difficulty
  stats$tooEasy = stats$difficulty > maxDifficulty

  # Lower limit on the item-rest correlation
  stats$weak = stats$itemRest < minItemRest

  # Items whose removal would raise alpha
  if (is.null(alpha)) {
    stats$hurtsAlpha = FALSE
  } else {
    stats$hurtsAlpha = stats$alphaIfDeleted > alpha
  }

  stats$flagged = stats$tooHard | stats$tooEasy | stats$weak | stats$hurtsAlpha
  stats
}

#' Distractor analysis for multiple-choice items
#'
#' @description Shows how often each option was chosen and by whom.
#' @param responses A character vector of chosen options for one item.
#' @param key The correct option.
#' @param total Numeric vector of total test scores, one per respondent.
#' @param options The answer options.
#' @return A data frame with columns `option`, `isKey`, `n`, `share`,
#'   `meanTotal` and `pointBiserial`, one row per option.
#' @export
distractorAnalysis = function(responses, key, total, options = c("A", "B", "C", "D")) {
  if (!key %in% options) {
    stop("`key` must be one of `options`.", call. = FALSE)
  }
  if (length(responses) != length(total)) {
    stop("`responses` and `total` must have the same length.", call. = FALSE)
  }
  out = data.frame(option = options, isKey = options == key)
  out$n = NA_integer_
  out$share = NA_real_
  out$meanTotal = NA_real_
  out$pointBiserial = NA_real_

  for (i in seq_along(options)) {
    chose = !is.na(responses) & responses == options[i]
    out$n[i] = sum(chose)
    out$share[i] = mean(chose)
    out$meanTotal[i] = if (any(chose)) mean(total[chose]) else NA_real_
    out$pointBiserial[i] = if (any(chose) && !all(chose)) {
      stats::cor(as.numeric(chose), total)
    } else {
      NA_real_
    }
  }

  out
}

#' Compare upper and lower scoring groups on each item
#'
#' @description Splits respondents by total score and contrasts the extremes.
#' @param items A numeric matrix or data frame of item scores, one column per
#'   item.
#' @param fraction Share of respondents in each extreme group.
#' @return A data frame with columns `item`, `upperMean`, `lowerMean`,
#'   `upperN`, `lowerN` and `index`.
#' @export
upperLowerIndex = function(items, fraction = 0.27) {
  if (fraction <= 0 || fraction > 0.5) {
    stop("`fraction` must lie in (0, 0.5].", call. = FALSE)
  }
  items = as.matrix(items)
  total = rowSums(items, na.rm = TRUE)
  ord = order(total)
  nGroup = max(1, floor(fraction * length(total)))

  # Upper group: highest totals
  upperRows = utils::tail(ord, nGroup)
  upperMean = colMeans(items[upperRows, , drop = FALSE], na.rm = TRUE)

  # Lower group: lowest totals
  lowerRows = utils::head(ord, nGroup)
  lowerMean = colMeans(items[lowerRows, , drop = FALSE], na.rm = TRUE)

  data.frame(
    item = colnames(items),
    upperMean = unname(upperMean),
    lowerMean = unname(lowerMean),
    upperN = length(upperRows),
    lowerN = length(lowerRows),
    index = unname(upperMean - lowerMean)
  )
}

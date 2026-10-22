#' Validate a row count for splitting
#'
#' Checks that `n` is a single positive whole number large enough to be
#' divided into `k` folds.
#'
#' @param n Number of rows to split.
#' @param k Number of folds.
#' @keywords internal
check_split_size <- function(n, k) {
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1) {
    stop("`n` must be a single positive number.", call. = FALSE)
  }
  if (!is.numeric(k) || length(k) != 1L || is.na(k) || k < 2) {
    stop("`k` must be a single number of at least 2.", call. = FALSE)
  }
  if (k > n) {
    stop(sprintf("Cannot make %d folds from %d rows.", as.integer(k), as.integer(n)),
      call. = FALSE)
  }
  return(invisible(as.integer(n)))
}

#' Build a split object from assessment indices
#'
#' Turns a list of held-out index vectors into a `fw_splits` object, where each
#' element stores both the analysis and assessment rows.
#'
#' @param holdout A list of integer vectors, one per fold, of assessment rows.
#' @param n Number of rows to split.
#' @param method A short label describing how the folds were made.
#' @keywords internal
new_splits <- function(holdout, n, method) {
  all_rows <- seq_len(n)
  folds <- vector("list", length(holdout))

  for (i in seq_along(holdout)) {
    assess <- sort(holdout[[i]])
    folds[[i]] <- list(
      id = sprintf("Fold%02d", i),
      analysis = setdiff(all_rows, assess),
      assessment = assess
    )
  }

  return(structure(folds, n = n, method = method, class = "fw_splits"))
}

#' Plain k-fold cross-validation splits
#'
#' Randomly assigns each row to one of `k` folds of nearly equal size.
#'
#' @param n Number of rows to split.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
#' @examples
#' kfold_splits(20, k = 4, seed = 1)
kfold_splits <- function(n, k = 5, seed = NULL) {
  n <- check_split_size(n, k)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # rep_len() keeps fold sizes within one row of each other
  fold_id <- sample(rep_len(seq_len(k), n))
  holdout <- split(seq_len(n), fold_id)
  return(new_splits(unname(holdout), n, method = "kfold"))
}

#' Stratified k-fold cross-validation splits
#'
#' Assigns rows to folds separately within each level of `strata`, so that
#' class proportions are roughly preserved in every fold.
#'
#' @param strata A vector of class labels, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
stratified_splits <- function(strata, k = 5, seed = NULL) {
  n <- check_split_size(length(strata), k)
  if (anyNA(strata)) {
    stop("`strata` must not contain missing values.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }

  fold_id <- integer(n)
  offset <- 0L
  for (level in unique(strata)) {
    rows <- which(strata == level)
    # rotate the starting fold so small classes do not all land in fold 1
    ids <- ((seq_along(rows) + offset - 1L) %% k) + 1L
    fold_id[rows] <- sample(ids)
    offset <- offset + length(rows)
  }

  holdout <- split(seq_len(n), fold_id)
  return(new_splits(unname(holdout), n, method = "stratified"))
}

#' Grouped k-fold cross-validation splits
#'
#' Keeps every row of a group in the same fold, so that no group appears in
#' both the analysis and assessment sets of a split.
#'
#' @param groups A vector of group identifiers, one per row.
#' @param k Number of folds.
#' @param seed Optional integer seed for reproducible assignment.
#' @return An object of class `fw_splits`.
#' @family splitters
#' @export
grouped_splits <- function(groups, k = 5, seed = NULL) {
  n <- length(groups)
  unique_groups <- unique(groups)
  check_split_size(length(unique_groups), k)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # greedy balancing: biggest groups first, each into the lightest fold;
  # the random second key breaks ties between equally sized groups
  sizes <- tabulate(match(groups, unique_groups), nbins = length(unique_groups))
  order_idx <- order(-sizes, sample(length(sizes)))
  load <- integer(k)
  group_fold <- integer(length(unique_groups))
  for (j in order_idx) {
    target <- which.min(load)
    group_fold[j] <- target
    load[target] <- load[target] + sizes[j]
  }

  fold_id <- group_fold[match(groups, unique_groups)]
  holdout <- split(seq_len(n), factor(fold_id, levels = seq_len(k)))
  return(new_splits(unname(holdout), n, method = "grouped"))
}

#' Leave-one-group-out splits
#'
#' Creates one split per distinct group, holding that group out entirely.
#'
#' @param groups A vector of group identifiers, one per row.
#' @return An object of class `fw_splits`.
#' @export
logo_splits <- function(groups) {
  n <- length(groups)
  unique_groups <- unique(groups)
  if (length(unique_groups) < 2L) {
    stop("Need at least two groups for leave-one-group-out.", call. = FALSE)
  }

  holdout <- lapply(unique_groups, function(g) which(groups %in% g))
  return(new_splits(holdout, n, method = "logo"))
}

#' Rolling-origin splits for ordered data
#'
#' Produces forward-chaining splits where each assessment window follows the
#' analysis window in time order.
#'
#' @param n Number of rows to split.
#' @param initial Number of rows in the first analysis window.
#' @param assess Number of rows in each assessment window.
#' @param cumulative If `TRUE`, the analysis window grows; otherwise it slides.
#' @return An object of class `fw_splits`.
#' @export
rolling_splits <- function(n, initial, assess = 1L, cumulative = TRUE) {
  if (initial < 1 || assess < 1) {
    stop("`initial` and `assess` must both be at least 1.", call. = FALSE)
  }
  if (initial + assess > n) {
    stop("`initial + assess` must not exceed `n`.", call. = FALSE)
  }

  starts <- seq.int(initial + 1L, n - assess + 1L, by = assess)
  folds <- vector("list", length(starts))
  for (i in seq_along(starts)) {
    # a sliding window keeps the analysis set at exactly `initial` rows
    first <- if (cumulative) 1L else starts[i] - initial
    folds[[i]] <- list(
      id = sprintf("Slice%03d", i),
      analysis = seq.int(first, starts[i] - 1L),
      assessment = seq.int(starts[i], starts[i] + assess - 1L)
    )
  }

  return(structure(folds, n = n, method = "rolling", class = "fw_splits"))
}

#' Print a set of splits
#'
#' Shows the splitting method and the analysis/assessment sizes of each fold.
#'
#' @param x An object of class `fw_splits`.
#' @param ... Ignored.
#' @export
print.fw_splits <- function(x, ...) {
  cat(sprintf("<fw_splits: %s, %d folds, n = %d>\n",
    attr(x, "method"), length(x), as.integer(attr(x, "n"))))
  for (fold in x) {
    cat(sprintf("  %-9s analysis %5d | assessment %5d\n",
      fold$id, length(fold$analysis), length(fold$assessment)))
  }
  return(invisible(x))
}

#' @title Validate treatment arms and allocation ratio
#' @description Checks that arm labels are unique and that the ratio has one
#'   positive whole number per arm.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @keywords internal
validateArms = function(arms, ratio) {
  if (anyDuplicated(arms)) {
    stop("Arm labels must be unique.", call. = FALSE)
  }
  badRatio = length(ratio) != length(arms) || anyNA(ratio) ||
    any(ratio < 1) || any(ratio != round(ratio))
  if (badRatio) {
    stop("`ratio` needs one positive whole number per arm.", call. = FALSE)
  }
  return(invisible(as.integer(ratio)))
}

#' @title Draw a single permuted block
#' @description Repeats each arm according to the allocation ratio until the
#'   block is filled, then shuffles the block.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @param blockSize Number of allocations in the block.
#' @return A character vector of length `blockSize`.
#' @keywords internal
permutedBlock = function(arms, ratio, blockSize) {
  ratio = validateArms(arms, ratio)
  unit = sum(ratio)
  if (blockSize %% unit != 0) {
    stop("`blockSize` must be a multiple of ", unit, ".", call. = FALSE)
  }

  block = rep(rep(arms, times = ratio), times = blockSize / unit)
  return(sample(block))
}

#' @title Block randomisation list
#' @description Generates a randomisation list from permuted blocks whose
#'   sizes are drawn at random from `blockSizes`, which makes the next
#'   allocation harder to guess.
#' @param n Minimum number of allocations to generate.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @param blockSizes Candidate block sizes, each a multiple of `sum(ratio)`.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param prefix Prefix for the participant allocation identifiers.
#' @return A data frame with columns `allocId`, `block`, `blockSize`,
#'   `position` and `arm`.
#' @export
blockRandomList = function(n, arms = c("A", "B"), ratio = c(1, 1),
                           blockSizes = c(4, 6), seed = NULL, prefix = "R") {
  if (length(n) != 1 || is.na(n) || n < 1) {
    stop("`n` must be a single positive number.", call. = FALSE)
  }
  ratio = validateArms(arms, ratio)
  if (any(blockSizes %% sum(ratio) != 0)) {
    stop("Every block size must be a multiple of ", sum(ratio), ".", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Keep adding whole blocks until n is covered; the last block is never cut
  # short, otherwise the final allocations would be predictable.
  blocks = list()
  total = 0
  b = 0
  while (total < n) {
    b = b + 1
    size = if (length(blockSizes) == 1) blockSizes else sample(blockSizes, 1)
    blocks[[b]] = data.frame(
      block = b,
      blockSize = size,
      position = seq_len(size),
      arm = permutedBlock(arms, ratio, size)
    )
    total = total + size
  }

  out = do.call(rbind, blocks)
  width = nchar(as.character(nrow(out)))
  out$allocId = sprintf(paste0("%s%0", width, "d"), prefix, seq_len(nrow(out)))
  out = out[, c("allocId", "block", "blockSize", "position", "arm")]
  attr(out, "seed") = seed
  return(out)
}

#' @title Stratified block randomisation list
#' @description Builds an independent block randomisation list within each
#'   stratum, so that balance holds inside every combination of
#'   stratification factors.
#' @param strata A data frame with one row per stratum and one column per
#'   stratification factor.
#' @param nPerStratum Minimum number of allocations per stratum.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param ... Passed to [blockRandomList()].
#' @return A data frame with the stratum columns followed by the
#'   [blockRandomList()] columns.
#' @export
stratifiedRandomList = function(strata, nPerStratum, seed = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  pieces = vector("list", nrow(strata))
  for (s in seq_len(nrow(strata))) {
    # as.character() so factor columns give their labels, not level codes
    values = vapply(strata[s, , drop = FALSE], as.character, character(1))
    label = paste(values, collapse = "-")
    lst = blockRandomList(nPerStratum, prefix = paste0(label, "-"), ...)
    pieces[[s]] = cbind(strata[rep(s, nrow(lst)), , drop = FALSE], lst)
  }

  out = do.call(rbind, pieces)
  rownames(out) = NULL
  attr(out, "seed") = seed
  return(out)
}

#' @title Simple (unrestricted) randomisation list
#' @description Independent draws with probabilities proportional to `ratio`;
#'   included mainly as a comparator when explaining why blocking is used.
#' @param n Number of allocations to generate.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param prefix Prefix for the participant allocation identifiers.
#' @return A data frame with columns `allocId` and `arm`.
#' @export
simpleRandomList = function(n, arms = c("A", "B"), ratio = c(1, 1), seed = NULL,
                            prefix = "R") {
  ratio = validateArms(arms, ratio)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  arm = sample(arms, n, replace = TRUE, prob = ratio / sum(ratio))
  width = nchar(as.character(n))
  out = data.frame(
    allocId = sprintf(paste0("%s%0", width, "d"), prefix, seq_len(n)),
    arm = arm
  )
  attr(out, "seed") = seed
  return(out)
}

#' @title Check the running balance of a randomisation list
#' @description Computes, for each allocation, the cumulative count per arm
#'   and the largest imbalance seen so far.
#' @param allocList A randomisation list with an `arm` column.
#' @param arms Character vector of treatment arm labels.
#' @return A list with the cumulative count matrix `counts` and the maximum
#'   absolute imbalance `maxImbalance`.
#' @export
checkBalance = function(allocList, arms = sort(unique(allocList$arm))) {
  counts = matrix(0L, nrow = nrow(allocList), ncol = length(arms))
  colnames(counts) = arms
  if (nrow(counts) == 0) {
    return(list(counts = counts, maxImbalance = 0L))
  }

  # %in% rather than == so a missing arm label counts for no arm instead of
  # turning every later cumulative count into NA
  for (a in arms) {
    counts[, a] = cumsum(allocList$arm %in% a)
  }
  spread = apply(counts, 1, function(row) max(row) - min(row))
  return(list(counts = counts, maxImbalance = max(spread)))
}

#' @title Mask arm labels with kit codes
#' @description Replaces treatment labels with opaque codes so that the list
#'   can be sent to the pharmacy without unblinding site staff.
#' @param allocList A randomisation list with an `arm` column.
#' @param codes A named character vector mapping arm labels to kit codes.
#' @return `allocList` with an added `kitCode` column and the `arm` column removed.
#' @export
maskArms = function(allocList, codes) {
  unknown = setdiff(unique(allocList$arm), names(codes))
  if (length(unknown) > 0) {
    stop("No kit code for arm(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }

  allocList$kitCode = unname(codes[allocList$arm])
  allocList$arm = NULL
  return(allocList)
}

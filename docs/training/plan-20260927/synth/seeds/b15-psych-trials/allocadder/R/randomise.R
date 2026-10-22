#' Validate treatment arms and allocation ratio
#'
#' Checks that arm labels are unique and that the ratio has one positive
#' whole number per arm.
#'
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @return The ratio as an integer vector, invisibly.
#' @keywords internal
validateArms = function(arms, ratio) {
  if (anyDuplicated(arms)) {
    stop("Arm labels must be unique.", call. = FALSE)
  }
  if (length(ratio) != length(arms) || any(ratio < 1) || any(ratio != round(ratio))) {
    stop("`ratio` needs one positive whole number per arm.", call. = FALSE)
  }
  return(invisible(as.integer(ratio)))
}

#' Draw a single permuted block
#'
#' Repeats each arm according to the allocation ratio until the block is
#' filled, then shuffles the block.
#'
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

#' Block randomisation list
#'
#' Generates a randomisation list from permuted blocks whose sizes are drawn
#' at random from `blockSizes`, which makes the next allocation harder to guess.
#'
#' @param n Minimum number of allocations to generate.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @param blockSizes Candidate block sizes, each a multiple of `sum(ratio)`.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param prefix Prefix for the participant allocation identifiers.
#' @return A data frame with columns `allocId`, `block`, `blockSize`,
#'   `position` and `arm`.
#' @family randomisation lists
#' @export
blockRandomList = function(n, arms = c("A", "B"), ratio = c(1, 1),
                           blockSizes = c(4, 6), seed = NULL, prefix = "R") {
  ratio = validateArms(arms, ratio)
  if (any(blockSizes %% sum(ratio) != 0)) {
    stop("Every block size must be a multiple of ", sum(ratio), ".", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
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

#' Stratified block randomisation list
#'
#' Builds an independent block randomisation list within each stratum, so
#' that balance holds inside every combination of stratification factors.
#'
#' @param strata A data frame with one row per stratum and one column per
#'   stratification factor.
#' @param nPerStratum Minimum number of allocations per stratum.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param ... Passed to [blockRandomList()].
#' @return A data frame with the stratum columns followed by the
#'   [blockRandomList()] columns.
#' @family randomisation lists
#' @export
stratifiedRandomList = function(strata, nPerStratum, seed = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  pieces = vector("list", nrow(strata))
  for (s in seq_len(nrow(strata))) {
    label = paste(unlist(strata[s, , drop = FALSE]), collapse = "-")
    lst = blockRandomList(nPerStratum, prefix = paste0(label, "-"), ...)
    pieces[[s]] = cbind(strata[rep(s, nrow(lst)), , drop = FALSE], lst)
  }
  out = do.call(rbind, pieces)
  rownames(out) = NULL
  attr(out, "seed") = seed
  return(out)
}

#' Simple (unrestricted) randomisation list
#'
#' Independent draws with probabilities proportional to `ratio`; included
#' mainly as a comparator when explaining why blocking is used.
#'
#' @param n Number of allocations to generate.
#' @param arms Character vector of treatment arm labels.
#' @param ratio Integer allocation ratio, one entry per arm.
#' @param seed Random seed recorded with the list for reproducibility.
#' @param prefix Prefix for the participant allocation identifiers.
#' @return A data frame with columns `allocId` and `arm`.
#' @family randomisation lists
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

#' Check the running balance of a randomisation list
#'
#' Computes, for each allocation, the cumulative count per arm and the
#' largest imbalance seen so far.
#'
#' @param allocList A randomisation list with an `arm` column.
#' @param arms Character vector of treatment arm labels.
#' @return A list with the cumulative count matrix `counts` and the maximum
#'   absolute imbalance `maxImbalance`.
#' @export
checkBalance = function(allocList, arms = sort(unique(allocList$arm))) {
  counts = matrix(0L, nrow = nrow(allocList), ncol = length(arms))
  colnames(counts) = arms
  for (a in arms) {
    counts[, a] = cumsum(allocList$arm == a)
  }
  spread = apply(counts, 1, function(row) max(row) - min(row))
  return(list(counts = counts, maxImbalance = max(spread)))
}

#' Mask arm labels with kit codes
#'
#' Replaces treatment labels with opaque codes so that the list can be sent
#' to the pharmacy without unblinding site staff.
#'
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

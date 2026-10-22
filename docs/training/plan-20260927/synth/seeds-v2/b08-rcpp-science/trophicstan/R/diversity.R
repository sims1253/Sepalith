#' Convert abundances to relative proportions
#'
#' @param counts A numeric vector of species abundances for one sample.
#' @keywords internal
as_proportions <- function(counts) {
  if (!is.numeric(counts) || any(counts < 0, na.rm = TRUE)) {
    stop("`counts` must be non-negative numbers.", call. = FALSE)
  }

  # Absent species contribute nothing and would give log(0) downstream
  counts <- counts[!is.na(counts) & counts > 0]
  if (length(counts) == 0L) {
    return(numeric())
  }
  counts / sum(counts)
}

#' Diversity indices for one sample
#'
#' `diversity_shannon()` is the Shannon entropy of the species proportions,
#' `diversity_simpson()` the Gini-Simpson index (between 0 and 1) and
#' `diversity_invsimpson()` the inverse Simpson index, the effective number
#' of common species.
#'
#' @param counts A numeric vector of species abundances for one sample.
#' @param base Base of the logarithm.
#' @return A single number, `NA` for an empty sample.
#' @export
diversity_shannon <- function(counts, base = exp(1)) {
  p <- as_proportions(counts)
  if (length(p) == 0L) {
    return(NA_real_)
  }
  -sum(p * log(p, base = base))
}

#' @rdname diversity_shannon
#' @export
diversity_simpson <- function(counts) {
  p <- as_proportions(counts)
  if (length(p) == 0L) {
    return(NA_real_)
  }
  1 - sum(p^2)
}

#' @rdname diversity_shannon
#' @export
diversity_invsimpson <- function(counts) {
  p <- as_proportions(counts)
  if (length(p) == 0L) {
    return(NA_real_)
  }
  1 / sum(p^2)
}

#' Chao1 species richness estimate
#'
#' Estimates total richness from the numbers of singletons and doubletons,
#' using the bias-corrected form when there are no doubletons.
#'
#' @param counts A numeric vector of species abundances for one sample.
#' @return A single number, the estimated species richness.
#' @export
richness_chao1 <- function(counts) {
  counts <- round(counts[!is.na(counts) & counts > 0])
  s_obs <- length(counts)
  if (s_obs == 0L) {
    return(0)
  }

  # Singletons and doubletons drive the estimate of unseen species
  f1 <- sum(counts == 1)
  f2 <- sum(counts == 2)
  n <- sum(counts)

  if (f2 > 0) {
    s_obs + (n - 1) / n * f1^2 / (2 * f2)
  } else {
    s_obs + (n - 1) / n * f1 * (f1 - 1) / 2
  }
}

#' Diversity indices for every sample in a community table
#'
#' @param community A sites-by-species matrix or data frame of abundances.
#' @param indices Character vector of indices to compute.
#' @return A data frame with a `sample` column and one column per index.
#' @export
community_diversity <- function(community,
                                indices = c("shannon", "simpson", "invsimpson", "chao1")) {
  indices <- match.arg(indices, several.ok = TRUE)
  community <- as.matrix(community)

  funs <- list(
    shannon = diversity_shannon,
    simpson = diversity_simpson,
    invsimpson = diversity_invsimpson,
    chao1 = richness_chao1
  )

  # Fall back to row numbers when the table has no sample names
  out <- data.frame(
    sample = rownames(community) %||% as.character(seq_len(nrow(community)))
  )
  for (idx in indices) {
    out[[idx]] <- unname(apply(community, 1, funs[[idx]]))
  }

  out
}

#' Lincoln-Petersen population estimate
#'
#' Two-sample mark-recapture estimate of closed population size.
#'
#' @param marked Number of animals marked in the first sample.
#' @param caught Number of animals caught in the second sample.
#' @param recaptured Number of marked animals in the second sample.
#' @return A single number, the estimated population size.
#' @family mark-recapture
#' @export
estimate_lincoln_petersen <- function(marked, caught, recaptured) {
  if (recaptured > min(marked, caught)) {
    stop("`recaptured` cannot exceed `marked` or `caught`.", call. = FALSE)
  }
  if (recaptured == 0) {
    return(Inf)
  }

  marked * caught / recaptured
}

#' Chapman population estimate
#'
#' Bias-corrected two-sample mark-recapture estimate of closed population size,
#' defined even when no marked animals are recaptured.
#'
#' @param marked Number of animals marked in the first sample.
#' @param caught Number of animals caught in the second sample.
#' @param recaptured Number of marked animals in the second sample.
#' @return A single number, the estimated population size.
#' @family mark-recapture
#' @export
estimate_chapman <- function(marked, caught, recaptured) {
  if (recaptured > min(marked, caught)) {
    stop("`recaptured` cannot exceed `marked` or `caught`.", call. = FALSE)
  }

  # Chapman's +1 terms keep the estimate finite when nothing is recaptured
  (marked + 1) * (caught + 1) / (recaptured + 1) - 1
}

callRelatedness <- function(
  pmr_tibble,
  class_prior = rep(0.25, 4),
  average_relatedness = NULL,
  median_co = 5e2,
  filter_n = 1
) {
  # Check pmr_tibble is of the form required:
  cols_check <- c('pair', 'nsnps', 'mismatch', 'pmr')
  if (!(all(cols_check %in% names(pmr_tibble)))) {
    stop(paste0(
      'Input tibble/data.frame must have 4 columns named: ',
      paste0(cols_check, collapse = ', '),
      '.'
    ))
  }

  # Check that the prior distribution makes sense:
  if (any(class_prior < 0) | any(class_prior > 1)) {
    stop(
      'Posterior probabilities for degrees of relatedness must be values between 0 and 1.'
    )
  } else if (abs(sum(class_prior) - 1) > 1e-3) {
    warning(
      'Posterior probabilities for degrees of relatedness must sum to 1. Normalising.'
    )
    class_prior <- class_prior / sum(class_prior)
  }

  # Check that the average relatedness makes sense:
  if (!is.null(average_relatedness)) {
    if (any(average_relatedness <= 0) | any(average_relatedness >= 1)) {
      stop('The average relatedness must be a value between 0 and 1.')
    }
  }

  # Check median estimator cut off:
  if ((median_co <= 0)) {
    stop(
      'The median cut off number of overlapping snps must be greater than 0.'
    )
  } else if (median_co >= max(pmr_tibble$nsnps, na.rm = T)) {
    stop(
      'The median cut off number of overlapping snps cannot be greater than the maximum number of overlapping SNPs that were observed!'
    )
  }

  # Check median estimator cut off:
  if ((filter_n <= 0)) {
    stop(
      'The cut off for the number of overlapping snps must be greater than 0.'
    )
  } else if (filter_n >= max(pmr_tibble$nsnps, na.rm = T)) {
    stop(
      'The cut off for the number of overlapping snps cannot be greater than the maximum number of overlapping SNPs that were observed!'
    )
  }

  # local variables
  class_vec <- c('Same_Twins', 'First_Degree', 'Second_Degree', 'Unrelated')

  # If no user-defined value for average_relatedness is given, use the median (above nsnps cut off)
  if (is.null(average_relatedness)) {
    M <- pmr_tibble %>%
      dplyr::filter(nsnps > median_co) %>%
      dplyr::pull(pmr) %>%
      stats::median(na.rm = T)
    filter_tibble <- pmr_tibble %>%
      dplyr::filter((nsnps >= filter_n)) %>%
      dplyr::rowwise() %>%
      dplyr::mutate(ave_rel = M)
  } else {
    M <- average_relatedness
    filter_tibble <- pmr_tibble %>%
      dplyr::mutate(ave_rel = M) %>%
      dplyr::filter((nsnps >= filter_n))
  }

  results_tibble <- filter_tibble %>%
    dplyr::ungroup() %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      Same_Twins = (stats::dbinom(mismatch, nsnps, 0.5 * ave_rel, log = T) +
        log(class_prior[1])),
      First_Degree = (stats::dbinom(
        mismatch,
        nsnps,
        (1 - 0.5^2) * ave_rel,
        log = T
      ) +
        log(class_prior[2])),
      Second_Degree = (stats::dbinom(
        mismatch,
        nsnps,
        (1 - 0.5^3) * ave_rel,
        log = T
      ) +
        log(class_prior[3])),
      Unrelated = weightedBinom(mismatch, nsnps, ave_rel, log = T) +
        log(class_prior[4])
    ) %>%
    dplyr::mutate(
      normConst = matrixStats::logSumExp(c(
        Same_Twins,
        First_Degree,
        Second_Degree,
        Unrelated
      )) %>%
        exp()
    ) %>%
    dplyr::mutate(
      relationship = class_vec[which.max(c(
        Same_Twins,
        First_Degree,
        Second_Degree,
        Unrelated
      ))]
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      Same_Twins = exp(Same_Twins) / normConst,
      First_Degree = exp(First_Degree) / normConst,
      Second_Degree = exp(Second_Degree) / normConst,
      Unrelated = exp(Unrelated) / normConst
    ) %>%
    dplyr::mutate(
      sd = sqrt(pmr * (1 - pmr) / nsnps),
      relationship = factor(
        relationship,
        levels = c('Same_Twins', 'First_Degree', 'Second_Degree', 'Unrelated')
      ),
      row = 1:dplyr::n()
    ) %>%
    dplyr::select(
      row,
      pair,
      relationship,
      pmr,
      sd,
      mismatch,
      nsnps,
      ave_rel,
      everything(),
      -normConst
    )
  return(results_tibble)
}
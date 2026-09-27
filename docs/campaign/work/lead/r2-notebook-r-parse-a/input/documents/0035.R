.truncateSimulationProfile <- function(profile, minCellCount = 5) {
  checkmate::assertClass(profile, "CohortDataSimulationProfile")
  checkmate::assertIntegerish(
    minCellCount,
    lower = 0L,
    upper = profile$metaData$populationSize
  )

  if (minCellCount == 0) {
    warning(
      "No truncation was done on low-prevalence covariates. Object may include low cell counts that enable identification of persons."
    )
    return(profile)
  }

  minObservedNonzeroPrevalence <- profile$covariatePrevalence |>
    filter(.data$prevalence > 0) |>
    summarize(min(.data$prevalence)) |>
    pull()

  message(sprintf(
    "Before truncating simulation profile, lowest non-zero covariate prevalence is %.08f (%.0f / %.0f)",
    minObservedNonzeroPrevalence,
    round(minObservedNonzeroPrevalence * profile$metaData$populationSize),
    profile$metaData$populationSize
  ))

  minimumAllowedPrevalence <- minCellCount / profile$metaData$populationSize

  profile$covariatePrevalence <- profile$covariatePrevalence |>
    mutate(
      prevalence = ifelse(
        .data$prevalence < minimumAllowedPrevalence,
        0,
        .data$prevalence
      )
    )

  truncatedMinObservedNonzeroPrevalence <- profile$covariatePrevalence |>
    filter(.data$prevalence > 0) |>
    summarize(min(.data$prevalence)) |>
    pull()

  message(sprintf(
    "After truncating simulation profile, lowest non-zero covariate prevalence is %.08f (%.0f / %.0f)",
    truncatedMinObservedNonzeroPrevalence,
    round(
      truncatedMinObservedNonzeroPrevalence * profile$metaData$populationSize
    ),
    profile$metaData$populationSize
  ))

  return(profile)
}
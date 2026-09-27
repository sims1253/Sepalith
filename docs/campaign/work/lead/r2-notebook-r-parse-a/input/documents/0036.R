createCohortMethodDataSimulationProfile <- function(
  cohortMethodData,
  minCellCount = 5
) {
  errorMessages <- checkmate::makeAssertCollection()
  checkmate::assertClass(
    cohortMethodData,
    "CohortMethodData",
    add = errorMessages
  )
  checkmate::reportAssertions(collection = errorMessages)

  if (nrow_temp(cohortMethodData$cohorts) == 0) {
    stop("Cohorts are empty")
  }

  if (nrow_temp(cohortMethodData$covariates) == 0) {
    stop("Covariates are empty")
  }

  if (
    sum(cohortMethodData$cohorts |> select("daysToCohortEnd") |> pull()) == 0
  ) {
    warning(
      "Cohort data appears to be limited, check daysToCohortEnd which appears to be all zeros"
    )
  }

  message("Computing covariate prevalence")
  # (Note: currently limiting to binary covariates)
  populationSize <- cohortMethodData$cohorts |>
    count() |>
    pull()
  covariatePrevalence <- cohortMethodData$covariates |>
    group_by(.data$covariateId) |>
    summarise(sum = sum(.data$covariateValue, na.rm = TRUE)) |>
    mutate(prevalence = .data$sum / populationSize) |>
    ungroup() |>
    inner_join(cohortMethodData$covariateRef, by = "covariateId") |>
    inner_join(cohortMethodData$analysisRef, by = "analysisId") |>
    filter(.data$isBinary == "Y") |>
    select("covariateId", "prevalence") |>
    collect()

  message("Computing propensity model")
  propensityScore <- createPs(
    cohortMethodData,
    createPsArgs = createCreatePsArgs(
      maxCohortSizeForFitting = 25000,
      prior = Cyclops::createPrior("laplace", 0.1, exclude = 0)
    )
  )
  propensityModel <- attr(propensityScore, "metaData")$psModelCoef

  message("Fitting outcome model(s)")
  outcomeIds <- attr(cohortMethodData, "metaData")$outcomeIds
  outcomeModels <- vector("list", length(outcomeIds))
  for (i in seq_along(outcomeIds)) {
    outcomeId <- outcomeIds[i]
    studyPop <- createStudyPopulation(
      cohortMethodData = cohortMethodData,
      population = propensityScore,
      outcomeId = outcomeId,
      createStudyPopulationArgs = createCreateStudyPopulationArgs(
        minDaysAtRisk = 1,
        removeSubjectsWithPriorOutcome = FALSE
      )
    )
    studyPop <- matchOnPs(
      population = studyPop,
      matchOnPsArgs = createMatchOnPsArgs(
        caliper = 0.25,
        caliperScale = "standardized",
        maxRatio = 1
      )
    )
    outcomeModel <- fitOutcomeModel(
      population = studyPop,
      cohortMethodData = cohortMethodData,
      fitOutcomeModelArgs = createFitOutcomeModelArgs(
        modelType = "poisson",
        stratified = FALSE,
        useCovariates = TRUE,
        prior = Cyclops::createPrior("laplace", 0.1, exclude = 0),
        control = Cyclops::createControl(threads = 2),
        profileBounds = NULL
      )
    )
    outcomeModels[[i]] <- outcomeModel$outcomeModelCoefficients[
      outcomeModel$outcomeModelCoefficients != 0
    ]
  }

  message("Computing rates of prior outcomes")
  totalTime <- cohortMethodData$cohorts |>
    summarise(time = sum(.data$daysFromObsStart, na.rm = TRUE)) |>
    pull()

  preIndexOutcomeRates <- cohortMethodData$outcomes |>
    filter(.data$daysToEvent < 0) |>
    group_by(.data$outcomeId) |>
    summarise(n = n_distinct(.data$rowId)) |>
    mutate(rate = .data$n / totalTime) |>
    select("outcomeId", "rate") |>
    collect()

  message(
    "Fitting models for time to observation period start, end and time to cohort end"
  )
  cohorts <- cohortMethodData$cohorts |>
    collect()

  obsEnd <- cohorts$daysToObsEnd
  cohortEnd <- cohorts$daysToCohortEnd
  event <- as.integer(cohortEnd < obsEnd)
  time <- cohortEnd
  time[cohortEnd > obsEnd] <- obsEnd[cohortEnd > obsEnd]
  data <- tibble(time = time, event = event)
  data <- data[data$time > 0, ]
  fitCohortEnd <- survival::survreg(
    survival::Surv(
      time,
      event
    ) ~ 1,
    data = data,
    dist = "exponential"
  )
  fitObsEnd <- survival::survreg(
    survival::Surv(obsEnd[obsEnd > 0]) ~ 1,
    dist = "exponential"
  )
  data <- cohorts
  minObsTime <- min(data$daysFromObsStart)
  data$time <- data$daysFromObsStart - minObsTime + 1
  fitObsStart <- survival::survreg(
    survival::Surv(time) ~ 1,
    data = data,
    dist = "exponential"
  )

  result <- list(
    covariatePrevalence = covariatePrevalence,
    propensityModel = propensityModel,
    outcomeModels = outcomeModels,
    preIndexOutcomeRates = preIndexOutcomeRates,
    metaData = attr(cohortMethodData, "metaData"),
    covariateRef = collect(cohortMethodData$covariateRef),
    analysisRef = collect(cohortMethodData$analysisRef),
    cohortEndRate = 1 / exp(coef(fitCohortEnd)),
    obsStartRate = 1 / exp(coef(fitObsStart)),
    minObsTime = minObsTime,
    obsEndRate = 1 / exp(coef(fitObsEnd))
  )

  class(result) <- "CohortDataSimulationProfile"
  result <- .truncateSimulationProfile(result, minCellCount)
  return(result)
}
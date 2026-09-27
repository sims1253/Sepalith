simulateCohortMethodData <- function(profile, n = 10000) {
  errorMessages <- checkmate::makeAssertCollection()
  checkmate::assertClass(
    profile,
    "CohortDataSimulationProfile",
    add = errorMessages
  )
  checkmate::assertInt(n, lower = 1, add = errorMessages)
  checkmate::reportAssertions(collection = errorMessages)

  message("Generating covariates")
  # Treatment variable is generated elsewhere:
  covariatePrevalence <- profile$covariatePrevalence[
    names(profile$covariatePrevalence) != "1"
  ]

  personsPerCov <- rpois(
    n = nrow(covariatePrevalence),
    lambda = covariatePrevalence$prevalence * n
  )
  personsPerCov[personsPerCov > n] <- n
  rowId <- sapply(personsPerCov, function(x, n) sample.int(size = x, n), n = n)
  rowId <- do.call("c", rowId)
  covariateIds <- covariatePrevalence$covariateId
  covariateId <- sapply(
    seq_along(personsPerCov),
    function(x, personsPerCov, covariateIds) {
      rep(
        covariateIds[x],
        personsPerCov[x]
      )
    },
    personsPerCov = personsPerCov,
    covariateIds = covariateIds
  )
  covariateId <- do.call("c", covariateId)
  covariateValue <- rep(1, length(covariateId))
  covariates <- tibble(
    rowId = rowId,
    covariateId = covariateId,
    covariateValue = covariateValue
  )

  message("Generating treatment variable")
  betas <- profile$propensityModel
  intercept <- betas[1]
  betas <- betas[2:length(betas)]
  betas <- tibble(
    beta = as.numeric(betas),
    covariateId = as.numeric(names(betas))
  )
  treatmentVar <- covariates |>
    inner_join(betas, by = "covariateId") |>
    mutate(value = .data$covariateValue * .data$beta) |>
    group_by(.data$rowId) |>
    summarise(value = sum(.data$value) + intercept)
  link <- function(x) {
    return(1 / (1 + exp(-x)))
  }
  treatmentVar$value <- link(treatmentVar$value)
  treatmentVar$rand <- runif(nrow(treatmentVar))
  treatmentVar$covariateValue <- as.integer(
    treatmentVar$rand < treatmentVar$value
  )
  treatmentVar <- treatmentVar[, c("rowId", "covariateValue")]
  treatmentVar$covariateId <- 1

  message("Generating cohorts")
  cohorts <- tibble(
    rowId = treatmentVar$rowId,
    treatment = treatmentVar$covariateValue,
    personId = treatmentVar$rowId,
    personSeqId = treatmentVar$rowId,
    cohortStartDate = as.Date("2000-01-01"),
    daysFromObsStart = profile$minObsTime +
      round(rexp(
        n,
        profile$obsStartRate
      )) -
      1,
    daysToCohortEnd = round(rexp(n, profile$obsEndRate)),
    daysToObsEnd = round(rexp(n, profile$cohortEndRate))
  )

  message("Generating outcomes after index date")
  allOutcomes <- tibble()
  for (i in seq_along(profile$metaData$outcomeIds)) {
    betas <- profile$outcomeModels[[i]]
    intercept <- betas[1]
    betas <- betas[2:length(betas)]
    betas <- tibble(
      beta = as.numeric(betas),
      covariateId = as.numeric(names(betas))
    )
    temp <- merge(covariates, betas)
    temp$value <- temp$covariateValue * temp$beta # Currently pointless, since covariateValue is always 1
    temp <- aggregate(value ~ rowId, data = temp, sum)
    temp$value <- temp$value + intercept
    temp$value <- exp(temp$value) # Value is now the rate
    temp <- merge(temp, cohorts[, c("rowId", "daysToObsEnd")])
    temp$value <- temp$value * temp$daysToObsEnd # Value is lambda
    temp$nOutcomes <- rpois(n, temp$value)
    temp$nOutcomes[temp$nOutcomes > temp$daysToObsEnd] <- temp$daysToObsEnd[
      temp$nOutcomes > temp$daysToObsEnd
    ]
    outcomeRows <- sum(temp$nOutcomes)
    outcomes <- tibble(
      rowId = rep(0, outcomeRows),
      outcomeId = rep(profile$metaData$outcomeIds[i], outcomeRows),
      daysToEvent = rep(0, outcomeRows)
    )
    cursor <- 1
    for (j in seq_len(nrow(temp))) {
      nOutcomes <- temp$nOutcomes[j]
      if (nOutcomes != 0) {
        outcomes$rowId[cursor:(cursor + nOutcomes - 1)] <- temp$rowId[j]
        outcomes$daysToEvent[cursor:(cursor + nOutcomes - 1)] <- sample.int(
          size = nOutcomes,
          temp$daysToObsEnd[j]
        )
        cursor <- cursor + nOutcomes
      }
    }
    allOutcomes <- rbind(allOutcomes, outcomes)
  }

  message("Generating outcomes before index date")
  for (i in seq_along(profile$metaData$outcomeIds)) {
    outcomeId <- profile$metaData$outcomeIds[i]
    rate <- profile$preIndexOutcomeRates$rate[
      profile$preIndexOutcomeRates$outcomeId == outcomeId
    ]
    nOutcomes <- rpois(nrow(cohorts), rate * cohorts$daysFromObsStart)
    nOutcomes[nOutcomes > cohorts$daysFromObsStart] <- cohorts$daysFromObsStart[
      nOutcomes > cohorts$daysFromObsStart
    ]
    outcomeRows <- sum(nOutcomes)
    outcomes <- tibble(
      rowId = rep(0, outcomeRows),
      outcomeId = rep(outcomeId, outcomeRows),
      daysToEvent = rep(0, outcomeRows)
    )
    cursor <- 1
    for (j in seq_along(nOutcomes)) {
      if (nOutcomes[j] != 0) {
        outcomes$rowId[cursor:(cursor + nOutcomes[j] - 1)] <- cohorts$rowId[j]
        outcomes$daysToEvent[cursor:(cursor + nOutcomes[j] - 1)] <- -sample.int(
          size = nOutcomes[j],
          cohorts$daysFromObsStart[j]
        )
        cursor <- cursor + nOutcomes[j]
      }
    }
    allOutcomes <- rbind(allOutcomes, outcomes)
  }

  result <- Andromeda::andromeda(
    outcomes = allOutcomes,
    cohorts = cohorts,
    covariates = covariates,
    covariateRef = profile$covariateRef,
    analysisRef = profile$analysisRef
  )
  metaData <- profile$metaData
  metaData$populationSize <- n
  attr(result, "metaData") <- metaData
  class(result) <- "CohortMethodData"
  attr(class(result), "package") <- "CohortMethod"
  return(result)
}
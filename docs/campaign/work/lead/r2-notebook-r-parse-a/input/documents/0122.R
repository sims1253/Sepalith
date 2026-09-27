createDechallengeRechallengeSettings <- function(
  targetIds,
  outcomeIds,
  dechallengeStopInterval = 30,
  dechallengeEvaluationWindow = 30
) {
  errorMessages <- checkmate::makeAssertCollection()
  # check targetIds is a vector of int/double
  .checkCohortIds(
    cohortIds = targetIds,
    type = "target",
    errorMessages = errorMessages
  )
  # check outcomeIds is a vector of int/double
  .checkCohortIds(
    cohortIds = outcomeIds,
    type = "outcome",
    errorMessages = errorMessages
  )

  # check dechallengeStopInterval is numeric
  checkmate::assertNumeric(
    x = dechallengeStopInterval,
    lower = 0,
    finite = TRUE,
    any.missing = FALSE,
    len = 1,
    .var.name = "dechallengeStopInterval",
    add = errorMessages
  )

  # check dechallengeEvaluationWindowl is numeric
  checkmate::assertNumeric(
    x = dechallengeEvaluationWindow,
    lower = 0,
    finite = TRUE,
    any.missing = FALSE,
    len = 1,
    .var.name = "dechallengeEvaluationWindow",
    add = errorMessages
  )

  checkmate::reportAssertions(errorMessages)

  # create data.frame with all combinations
  result <- list(
    targetCohortDefinitionIds = targetIds,
    outcomeCohortDefinitionIds = outcomeIds,
    dechallengeStopInterval = dechallengeStopInterval,
    dechallengeEvaluationWindow = dechallengeEvaluationWindow
  )

  class(result) <- "dechallengeRechallengeSettings"
  return(result)
}
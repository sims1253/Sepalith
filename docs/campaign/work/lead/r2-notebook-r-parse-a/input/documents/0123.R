getDechallengeRechallengeJobs <- function(
  characterizationSettings,
  nTargetJobs
) {
  characterizationSettings <- characterizationSettings$dechallengeRechallengeSettings
  if (length(characterizationSettings) == 0) {
    return(NULL)
  }
  ind <- seq_along(characterizationSettings)
  targetIds <- lapply(ind, function(i) {
    characterizationSettings[[i]]$targetCohortDefinitionIds
  })
  outcomeIds <- lapply(ind, function(i) {
    characterizationSettings[[i]]$outcomeCohortDefinitionIds
  })
  dechallengeStopIntervals <- lapply(ind, function(i) {
    characterizationSettings[[i]]$dechallengeStopInterval
  })
  dechallengeEvaluationWindows <- lapply(ind, function(i) {
    characterizationSettings[[i]]$dechallengeEvaluationWindow
  })

  # get all combinations of TnOs, then split by treads

  combinations <- do.call(
    what = "rbind",
    args = lapply(
      seq_along(targetIds),
      function(i) {
        result <- expand.grid(
          targetId = targetIds[[i]],
          outcomeId = outcomeIds[[i]]
        )
        result$dechallengeStopInterval <- dechallengeStopIntervals[[i]]
        result$dechallengeEvaluationWindow <- dechallengeEvaluationWindows[[i]]
        return(result)
      }
    )
  )
  # find out whether more Ts or more Os
  tcount <- nrow(
    combinations %>%
      dplyr::count(
        .data$targetId,
        .data$dechallengeStopInterval,
        .data$dechallengeEvaluationWindow
      )
  )

  ocount <- nrow(
    combinations %>%
      dplyr::count(
        .data$outcomeId,
        .data$dechallengeStopInterval,
        .data$dechallengeEvaluationWindow
      )
  )

  if (nTargetJobs > max(tcount, ocount)) {
    message(
      "Input parameter nTargetJobs greater than number of targets and outcomes"
    )
    message(paste0(
      "Only using ",
      max(tcount, ocount),
      " nTargetJobs for DechallengeRechallenge"
    ))
  }

  if (tcount >= ocount) {
    threadDf <- combinations %>%
      dplyr::count(
        .data$targetId,
        .data$dechallengeStopInterval,
        .data$dechallengeEvaluationWindow
      )
    threadDf$nTargetJobs <- rep(1:nTargetJobs, ceiling(tcount / nTargetJobs))[
      1:tcount
    ]
    mergeColumn <- c(
      "targetId",
      "dechallengeStopInterval",
      "dechallengeEvaluationWindow"
    )
  } else {
    threadDf <- combinations %>%
      dplyr::count(
        .data$outcomeId,
        .data$dechallengeStopInterval,
        .data$dechallengeEvaluationWindow
      )
    threadDf$nTargetJobs <- rep(1:nTargetJobs, ceiling(ocount / nTargetJobs))[
      1:ocount
    ]
    mergeColumn <- c(
      "outcomeId",
      "dechallengeStopInterval",
      "dechallengeEvaluationWindow"
    )
  }

  combinations <- merge(combinations, threadDf, by = mergeColumn)
  sets <- lapply(
    X = 1:max(threadDf$nTargetJobs),
    FUN = function(i) {
      createDechallengeRechallengeSettings(
        targetIds = unique(combinations$targetId[
          combinations$nTargetJobs == i
        ]),
        outcomeIds = unique(combinations$outcomeId[
          combinations$nTargetJobs == i
        ]),
        dechallengeStopInterval = unique(combinations$dechallengeStopInterval[
          combinations$nTargetJobs == i
        ]),
        dechallengeEvaluationWindow = unique(combinations$dechallengeEvaluationWindow[
          combinations$nTargetJobs == i
        ])
      )
    }
  )

  # recreate settings
  settings <- c()
  for (i in seq_along(sets)) {
    settings <- rbind(
      settings,
      data.frame(
        functionName = "computeDechallengeRechallengeAnalyses",
        settings = as.character(ParallelLogger::convertSettingsToJson(
          sets[[i]]
        )),
        executionFolder = paste0("dr_", i),
        jobId = paste0("dr_", i)
      )
    )
    settings <- rbind(
      settings,
      data.frame(
        functionName = "computeRechallengeFailCaseSeriesAnalyses",
        settings = as.character(ParallelLogger::convertSettingsToJson(
          sets[[i]]
        )),
        executionFolder = paste0("rfcs_", i),
        jobId = paste0("rfcs_", i)
      )
    )
  }

  return(settings)
}
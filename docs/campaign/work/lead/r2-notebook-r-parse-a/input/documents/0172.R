checkDaysSupply <- function(
  cdm,
  drugRecordsTable = "ingredient_drug_records",
  byConcept = TRUE,
  sampleSize = 10000
) {
  errorMessage <- checkmate::makeAssertCollection()
  checkDbType(cdm = cdm, messageStore = errorMessage)
  checkTableExists(
    cdm = cdm,
    tableName = drugRecordsTable,
    messageStore = errorMessage
  )
  checkmate::reportAssertions(collection = errorMessage)

  records <- cdm[[drugRecordsTable]] %>%
    dplyr::select(
      "drug_concept_id",
      "drug",
      "ingredient_concept_id",
      "ingredient",
      "drug_exposure_start_date",
      "drug_exposure_end_date",
      "days_supply",
      "person_id"
    )

  records <- records %>%
    dplyr::mutate(
      n = dplyr::if_else(
        days_supply ==
          as.integer(
            !!CDMConnector::datediff(
              start = "drug_exposure_start_date",
              end = "drug_exposure_end_date",
              interval = "day"
            )
          ),
        0,
        1
      )
    ) %>%
    dplyr::collect()

  if (isTRUE(byConcept)) {
    grouping <- c(
      "drug_concept_id",
      "drug",
      "ingredient_concept_id",
      "ingredient"
    )
  } else {
    grouping <- c("ingredient_concept_id", "ingredient")
  }

  records <- records %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(grouping))) %>%
    dplyr::summarise(
      n_records = as.integer(dplyr::n()),
      n_sample = .env$sampleSize,
      n_different_days_supply_and_drug_dates = sum(.data$n, na.rm = T),
      n_days_supply_match_drug_dates = sum(.data$n == 0, na.rm = T),
      n_missing_days_supply = sum(is.na(.data$n)),
      proportion_different_days_supply_and_drug_dates = .data$n_different_days_supply_and_drug_dates /
        dplyr::n(),
      proportion_days_supply_match_drug_dates = .data$n_days_supply_match_drug_dates /
        dplyr::n(),
      proportion_missing_days_supply = .data$n_missing_days_supply / dplyr::n(),
      minimum_drug_exposure_days_supply = min(.data$days_supply, na.rm = T),
      maximum_drug_exposure_days_supply = max(.data$days_supply, na.rm = T),
      n_person = dplyr::n_distinct(.data$person_id),
      q05_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.05,
        na.rm = T
      ),
      q10_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.10,
        na.rm = T
      ),
      q25_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.25,
        na.rm = T
      ),
      median_drug_exposure_days_supply = stats::median(
        .data$days_supply,
        na.rm = T
      ),
      q75_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.75,
        na.rm = T
      ),
      q90_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.90,
        na.rm = T
      ),
      q95_drug_exposure_days_supply = stats::quantile(
        .data$days_supply,
        0.95,
        na.rm = T
      ),
    ) %>%
    dplyr::select(dplyr::all_of(c(
      grouping,
      "n_records",
      "n_sample",
      "n_person",
      "minimum_drug_exposure_days_supply",
      "q05_drug_exposure_days_supply",
      "q10_drug_exposure_days_supply",
      "q25_drug_exposure_days_supply",
      "median_drug_exposure_days_supply",
      "q75_drug_exposure_days_supply",
      "q90_drug_exposure_days_supply",
      "q95_drug_exposure_days_supply",
      "maximum_drug_exposure_days_supply",
      "n_different_days_supply_and_drug_dates",
      "n_days_supply_match_drug_dates",
      "n_missing_days_supply",
      "proportion_different_days_supply_and_drug_dates",
      "proportion_days_supply_match_drug_dates",
      "proportion_missing_days_supply"
    )))
  return(records)
}
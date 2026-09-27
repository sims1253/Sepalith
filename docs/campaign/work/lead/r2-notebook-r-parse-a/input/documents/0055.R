cdmFlatten <- function(
  cdm,
  domain = c("condition_occurrence", "drug_exposure", "procedure_occurrence"),
  includeConceptName = TRUE
) {
  checkmate::assertClass(cdm, "cdm_reference")
  checkmate::assertCharacter(domain, min.len = 1)
  checkmate::assertSubset(
    domain,
    choices = c(
      "condition_occurrence",
      "drug_exposure",
      "procedure_occurrence",
      "measurement",
      "visit_occurrence",
      "death",
      "observation"
    )
  )

  checkmate::assertLogical(includeConceptName, len = 1)

  queryList <- list()

  if ("condition_occurrence" %in% domain) {
    checkmate::assertTRUE("condition_occurrence" %in% names(cdm))
    queryList[["condition_occurrence"]] <- cdm$condition_occurrence %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$condition_concept_id,
        start_date = .data$condition_start_date,
        end_date = .data$condition_end_date,
        type_concept_id = .data$condition_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "condition_occurrence")
  }

  if ("drug_exposure" %in% domain) {
    checkmate::assertTRUE("drug_exposure" %in% names(cdm))
    queryList[["drug_exposure"]] <- cdm$drug_exposure %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$drug_concept_id,
        start_date = .data$drug_exposure_start_date,
        end_date = .data$drug_exposure_end_date,
        type_concept_id = .data$drug_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "drug_exposure")
  }

  if ("procedure_occurrence" %in% domain) {
    checkmate::assertTRUE("procedure_occurrence" %in% names(cdm))
    queryList[["procedure_occurrence"]] <- cdm$procedure_occurrence %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$procedure_concept_id,
        start_date = .data$procedure_date,
        end_date = .data$procedure_date,
        type_concept_id = .data$procedure_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "procedure_occurrence")
  }

  if ("measurement" %in% domain) {
    checkmate::assertTRUE("measurement" %in% names(cdm))
    queryList[["measurement"]] <- cdm$measurement %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$measurement_concept_id,
        start_date = .data$measurement_date,
        end_date = .data$measurement_date,
        type_concept_id = .data$measurement_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "measurement")
  }

  if ("visit_occurrence" %in% domain) {
    checkmate::assertTRUE("visit_occurrence" %in% names(cdm))
    queryList[["visit_occurrence"]] <- cdm$visit_occurrence %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$visit_concept_id,
        start_date = .data$visit_start_date,
        end_date = .data$visit_end_date,
        type_concept_id = .data$visit_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "visit_occurrence")
  }

  if ("death" %in% domain) {
    checkmate::assertTRUE("death" %in% names(cdm))
    queryList[["death"]] <- cdm$death %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$cause_concept_id,
        start_date = .data$death_date,
        end_date = .data$death_date,
        type_concept_id = .data$death_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "death")
  }

  if ("observation" %in% domain) {
    checkmate::assertTRUE("observation" %in% names(cdm))
    queryList[["death"]] <- cdm$observation %>%
      dplyr::transmute(
        person_id = .data$person_id,
        observation_concept_id = .data$observation_concept_id,
        start_date = .data$observation_date,
        end_date = .data$observation_date,
        type_concept_id = .data$observation_type_concept_id
      ) %>%
      dplyr::distinct() %>%
      dplyr::mutate(domain = "observation")
  }

  if (includeConceptName) {
    checkmate::assertTRUE("concept" %in% names(cdm))
    out <- queryList %>%
      purrr::reduce(dplyr::union) %>%
      dplyr::left_join(
        dplyr::transmute(
          cdm$concept,
          observation_concept_id = .data$concept_id,
          observation_concept_name = .data$concept_name
        ),
        by = "observation_concept_id"
      ) %>%
      dplyr::left_join(
        dplyr::transmute(
          cdm$concept,
          type_concept_id = .data$concept_id,
          type_concept_name = .data$concept_name
        ),
        by = "type_concept_id"
      ) %>%
      dplyr::ungroup() %>%
      dplyr::distinct()
  } else {
    out <- purrr::reduce(queryList, dplyr::union) %>%
      dplyr::ungroup() %>%
      dplyr::distinct()
  }

  # compute?
  return(out)
}
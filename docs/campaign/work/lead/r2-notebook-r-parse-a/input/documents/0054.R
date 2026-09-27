cdmSamplePerson <- function(cdm, person_subset) {
  checkmate::assert_class(cdm, "cdm_reference")
  checkmate::assert_class(person_subset, "tbl_sql")

  for (nm in names(cdm)) {
    if ("person_id" %in% colnames(cdm[[nm]])) {
      cdm[[nm]] <- dplyr::inner_join(cdm[[nm]], person_subset, by = "person_id")
    } else if ("subject_id" %in% colnames(cdm[[nm]])) {
      cdm[[nm]] <- dplyr::inner_join(
        cdm[[nm]],
        person_subset,
        by = c("subject_id" = "person_id")
      )
    }
  }
  return(cdm)
}
gridding_internal <- function(meta, btf, res, resByData, verbose) {
  checkmate::assert_names(
    x = colnames(meta),
    what = "colnames",
    must.include = c(
      "STUDY_ID",
      "NUMBER_LAT_LONG",
      "AREA_SQ_KM",
      "CENT_LONG",
      "CENT_LAT",
      "REALM",
      "CLIMATE",
      "TAXA",
      "ABUNDANCE_TYPE",
      "BIOMASS_TYPE"
    )
  )
  checkmate::assert_names(
    x = colnames(btf),
    what = "colnames",
    must.include = c(
      "valid_name",
      "STUDY_ID",
      "DAY",
      "MONTH",
      "YEAR",
      "LATITUDE",
      "LONGITUDE",
      "ABUNDANCE",
      "BIOMASS",
      "SAMPLE_DESC",
      "resolution",
      "taxon"
    )
  )
  checkmate::assert_numeric(btf$ABUNDANCE, lower = 0)
  checkmate::assert_numeric(btf$BIOMASS, lower = 0)
  checkmate::assert_number(
    x = res,
    lower = 0,
    upper = 30,
    null.ok = FALSE,
    na.ok = FALSE
  )
  checkmate::assert_logical(
    resByData,
    len = 1L,
    any.missing = FALSE,
    null.ok = FALSE,
  )
  checkmate::assert_logical(
    verbose,
    len = 1L,
    any.missing = FALSE,
    null.ok = FALSE,
  )

  AREA_SQ_KM <- meta[
    meta$NUMBER_LAT_LONG == 1L & meta$AREA_SQ_KM <= 500,
    "AREA_SQ_KM"
  ] |>
    unlist()

  SL_extent <- sum(
    base::mean(AREA_SQ_KM, na.rm = TRUE),
    stats::sd(AREA_SQ_KM, na.rm = TRUE)
  )

  meta$StudyMethod <- data.table::fifelse(
    test = meta$NUMBER_LAT_LONG == 1L | meta$AREA_SQ_KM < SL_extent,
    yes = "SL",
    no = "ML"
  )

  bt <- dplyr::inner_join(
    x = meta |>
      dplyr::select(
        "STUDY_ID",
        "CLIMATE",
        "REALM",
        "TAXA",
        "StudyMethod",
        "CENT_LAT",
        "CENT_LONG",
        "ABUNDANCE_TYPE",
        "BIOMASS_TYPE"
      ),
    y = btf |>
      dplyr::select(
        "STUDY_ID",
        "SAMPLE_DESC",
        "taxon",
        "LATITUDE",
        "LONGITUDE",
        "YEAR",
        "MONTH",
        "DAY",
        Species = "valid_name",
        "resolution",
        "ABUNDANCE",
        "BIOMASS"
      ),
    by = dplyr::join_by("STUDY_ID")
  )

  data.table::setorder(bt, "StudyMethod")

  bt$lon_to_grid <- data.table::fifelse(
    test = bt$StudyMethod == "SL",
    yes = bt$CENT_LONG,
    no = bt$LONGITUDE
  )
  bt$lat_to_grid <- data.table::fifelse(
    test = bt$StudyMethod == "SL",
    yes = bt$CENT_LAT,
    no = bt$LATITUDE
  )

  # See benchmarks.R # counting one year studies
  one_year_studies <- tapply(bt$YEAR, bt$STUDY_ID, function(y) {
    data.table::uniqueN(y) == 1L
  })

  if (any(one_year_studies)) {
    # See benchmarks.R  # Row filtering ----
    bt <- bt |>
      dplyr::filter(
        !is.element(STUDY_ID, names(one_year_studies)[which(one_year_studies)])
      )
    if (verbose) warning("Some 1-year-long studies were removed.")
  }

  dgg <- dggridR::dgconstruct(res = res)

  if (resByData) {
    res <- dggridR::dg_closest_res_to_area(dgg, SL_extent)
    dgg <- dggridR::dgsetres(dgg, res)
  }

  bt$cell <- dggridR::dgGEO_to_SEQNUM(
    dggs = dgg,
    in_lon_deg = bt$lon_to_grid,
    in_lat_deg = bt$lat_to_grid
  )$seqnum

  bt$assemblageID <- base::paste(bt$STUDY_ID, bt$cell, sep = "_")
  bt <- bt |>
    dplyr::select(-c("CENT_LAT", "CENT_LONG", "lat_to_grid", "lon_to_grid"))

  if (
    tapply(
      X = bt$cell[bt$StudyMethod == "SL"],
      INDEX = bt$STUDY_ID[bt$StudyMethod == "SL"],
      FUN = function(x) data.table::uniqueN(x) == 1L
    ) |>
      all()
  ) {
    if (verbose) base::message("OK: all SL studies have 1 grid cell")
  } else {
    base::stop("ERROR: some SL studies have > 1 grid cell")
  }

  return(bt)
}
#' @title Read one site's visit export
#' @description Reads the CSV export of a single site, standardises column names and
#'   tags every row with the site identifier taken from the file name
#'   (`site_<id>_visits.csv`).
#' @param path Path to a site export file.
#' @return A tibble with a leading `site_id` column.
#' @export
read.site.file = function(path) {
  site_id = stringr::str_match(basename(path), "^site_([A-Za-z0-9]+)_")[, 2]
  if (is.na(site_id)) {
    stop("Cannot find a site id in file name '", basename(path), "'.", call. = FALSE)
  }

  # Everything is read as character; types are fixed in harmonise.types().
  readr::read_csv(path, col_types = readr::cols(.default = "c"), progress = FALSE) %>%
    dplyr::rename_with(~ stringr::str_replace_all(stringr::str_to_lower(.x), "\\s+", "_")) %>%
    dplyr::mutate(site_id = site_id, .before = 1)
}

#' @title Read and combine visit exports from every site
#' @description Files that fail to read are skipped with a warning naming them, so one
#'   corrupt export does not block the whole data cut.
#' @param dir Directory containing the site export files.
#' @param pattern Regular expression matching export file names.
#' @return A tibble of all sites' visits, with a `"failed_files"` attribute.
#' @export
read.all.sites = function(dir, pattern = "^site_.*_visits\\.csv$") {
  paths = list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(paths) == 0) {
    stop("No site exports found in '", dir, "'.", call. = FALSE)
  }

  results = paths %>%
    purrr::set_names(basename(paths)) %>%
    purrr::map(purrr::safely(read.site.file))
  errors = purrr::map(results, "error") %>% purrr::compact()
  if (length(errors) > 0) {
    warning("Skipped ", length(errors), " file(s): ",
      paste(names(errors), collapse = ", "), call. = FALSE)
  }

  combined = results %>%
    purrr::map("result") %>%
    purrr::compact() %>%
    purrr::list_rbind()
  attr(combined, "failed_files") = names(errors)
  combined
}

#' @title Harmonise column types across site exports
#' @description Applies a named list of type conversions, e.g.
#'   `list(visit_date = parse.visit.date, aval = as.numeric)`, to the columns
#'   that exist; missing columns are reported but not created.
#' @param data A data frame of combined site visits.
#' @param types A named list of conversion functions, one per column.
#' @return `data` with converted columns.
#' @export
harmonise.types = function(data, types) {
  absent = setdiff(names(types), names(data))
  if (length(absent) > 0) {
    message("Columns not found, skipping: ", paste(absent, collapse = ", "))
  }
  present = types[intersect(names(types), names(data))]

  # Apply one conversion per column, threading the data through.
  purrr::reduce2(
    names(present), present,
    function(acc, col, fn) dplyr::mutate(acc, dplyr::across(dplyr::all_of(col), fn)),
    .init = data
  )
}

#' @title Count visits per site
#' @param data A data frame of combined site visits.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site and an `n_visits` column.
#' @export
site.visit.counts = function(data, site = site_id) {
  data %>%
    dplyr::count({{ site }}, name = "n_visits") %>%
    dplyr::arrange(dplyr::desc(.data$n_visits))
}

#' @title Count subjects per site
#' @param data A data frame of combined site visits.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site and an `n_subjects` column.
#' @export
site.subject.counts = function(data, site = site_id) {
  data %>%
    dplyr::distinct({{ site }}, .data$subject_id) %>%
    dplyr::count({{ site }}, name = "n_subjects") %>%
    dplyr::arrange(dplyr::desc(.data$n_subjects))
}

#' @title Count out-of-window visits per site
#' @param data A data frame of combined site visits with `early` and `late`
#'   columns from [flag.early.visits()] and [flag.late.visits()].
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site and an `n_out_of_window` column.
#' @export
site.deviation.counts = function(data, site = site_id) {
  data %>%
    dplyr::filter(.data$early | .data$late) %>%
    dplyr::count({{ site }}, name = "n_out_of_window") %>%
    dplyr::arrange(dplyr::desc(.data$n_out_of_window))
}

#' @title Summarise data completeness per site
#' @description Computes the share of non-missing values in each requested column,
#'   per site, and an overall completeness score as their mean.
#' @param data A data frame of combined site visits.
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Columns to assess.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per site, one column per assessed column
#'   and an `overall` column, sorted from least to most complete.
#' @export
site.completeness = function(data, cols, site = site_id) {
  data %>%
    dplyr::group_by({{ site }}) %>%
    dplyr::summarise(dplyr::across({{ cols }}, ~ mean(!is.na(.x))), .groups = "drop") %>%
    dplyr::mutate(
      overall = rowMeans(dplyr::pick(-{{ site }})),
      {{ site }} := forcats::fct_reorder(as.character({{ site }}), .data$overall)
    ) %>%
    dplyr::arrange(.data$overall)
}

#' @title Build a data query listing for site monitors
#' @description Collects the issues site monitors follow up on (missing visit dates,
#'   out-of-window visits, duplicated visits and missing results) into one
#'   listing with a query text per row.
#' @param data A data frame of combined site visits.
#' @param result_cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Result columns that
#'   must be filled at every visit.
#' @param site <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the site.
#' @return A tibble with one row per query, sorted by site and subject.
#' @export
site.query.listing = function(data, result_cols, site = site_id) {
  keys = data %>%
    dplyr::select({{ site }}, "subject_id", "visit")

  # Each query type is built separately with the same key columns so the
  # pieces can simply be stacked at the end.
  missing.dates = data %>%
    dplyr::filter(is.na(.data$visit_date)) %>%
    dplyr::transmute({{ site }}, .data$subject_id, .data$visit,
      query = "Visit date is missing.")
  out.of.window = data %>%
    dplyr::filter(.data$early | .data$late) %>%
    dplyr::transmute({{ site }}, .data$subject_id, .data$visit,
      query = sprintf("Visit on study day %d is %s by %d day(s).",
        .data$study_day, dplyr::if_else(.data$early, "early", "late"),
        abs(.data$deviation)))
  duplicates = keys %>%
    dplyr::count({{ site }}, .data$subject_id, .data$visit, name = "n") %>%
    dplyr::filter(.data$n > 1) %>%
    dplyr::transmute({{ site }}, .data$subject_id, .data$visit,
      query = sprintf("Visit recorded %d times.", .data$n))

  # Pivot result columns long so that every missing cell becomes one query.
  missing.results = data %>%
    dplyr::select({{ site }}, "subject_id", "visit", {{ result_cols }}) %>%
    tidyr::pivot_longer(
      cols = -c({{ site }}, "subject_id", "visit"),
      names_to = "result",
      values_to = "value",
      values_transform = as.character
    ) %>%
    dplyr::filter(is.na(.data$value)) %>%
    dplyr::transmute({{ site }}, .data$subject_id, .data$visit,
      query = sprintf("Result '%s' is missing.", .data$result))

  list(missing.dates, out.of.window, duplicates, missing.results) %>%
    purrr::list_rbind() %>%
    dplyr::arrange({{ site }}, .data$subject_id, .data$visit) %>%
    dplyr::mutate(query_id = sprintf("Q%05d", dplyr::row_number()), .before = 1)
}

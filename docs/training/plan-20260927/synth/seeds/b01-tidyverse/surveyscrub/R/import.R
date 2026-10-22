#' Read a survey wave exported as CSV
#'
#' Reads a comma-separated survey export with every column as character so
#' that missing-value codes survive the import untouched.
#'
#' @param path Path to the exported survey file.
#' @param wave A single integer identifying the survey wave.
#' @param na Character vector of strings to treat as missing on import.
#' @return A tibble with cleaned column names and a `wave` column.
#' @family wave readers
#' @export
read_wave_csv <- function(path, wave, na = c("", "NA")) {
  check_wave_path(path)
  raw <- readr::read_csv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    na = na,
    progress = FALSE
  )
  raw %>%
    rename_survey_cols() %>%
    dplyr::mutate(wave = as.integer(wave), .before = 1)
}

#' Read a survey wave exported as TSV
#'
#' Reads a tab-separated survey export with every column as character so
#' that missing-value codes survive the import untouched.
#'
#' @param path Path to the exported survey file.
#' @param wave A single integer identifying the survey wave.
#' @param na Character vector of strings to treat as missing on import.
#' @return A tibble with cleaned column names and a `wave` column.
#' @family wave readers
#' @export
read_wave_tsv <- function(path, wave, na = c("", "NA")) {
  check_wave_path(path)
  raw <- readr::read_tsv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    na = na,
    progress = FALSE
  )
  raw %>%
    rename_survey_cols() %>%
    dplyr::mutate(wave = as.integer(wave), .before = 1)
}

#' Read a survey wave exported as semicolon-delimited text
#'
#' Reads a semicolon-separated export (common for European survey tools,
#' which use a decimal comma) with every column as character.
#'
#' @param path Path to the exported survey file.
#' @param wave A single integer identifying the survey wave.
#' @param na Character vector of strings to treat as missing on import.
#' @return A tibble with cleaned column names and a `wave` column.
#' @family wave readers
#' @export
read_wave_csv2 <- function(path, wave, na = c("", "NA")) {
  check_wave_path(path)
  raw <- readr::read_csv2(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    na = na,
    progress = FALSE
  )
  raw %>%
    rename_survey_cols() %>%
    dplyr::mutate(wave = as.integer(wave), .before = 1)
}

#' Read any supported survey wave export
#'
#' Dispatches to the matching reader based on the requested format, or on
#' the file extension when `format` is left at its default.
#'
#' @param path Path to the exported survey file.
#' @param wave A single integer identifying the survey wave.
#' @param format One of `"auto"`, `"csv"`, `"tsv"` or `"csv2"`.
#' @param ... Passed to [read_wave_csv()], [read_wave_tsv()] or [read_wave_csv2()].
#' @return A tibble with cleaned column names and a `wave` column.
#' @family wave readers
#' @export
read_wave <- function(path, wave, format = c("auto", "csv", "tsv", "csv2"), ...) {
  format <- match.arg(format)
  if (format == "auto") {
    ext <- tolower(tools::file_ext(path))
    format <- switch(ext,
      csv = "csv",
      tsv = ,
      txt = "tsv",
      stop("Cannot guess the format of '", basename(path), "'.", call. = FALSE)
    )
  }
  switch(format,
    csv = read_wave_csv(path, wave, ...),
    tsv = read_wave_tsv(path, wave, ...),
    csv2 = read_wave_csv2(path, wave, ...)
  )
}

#' Stack several survey waves into one table
#'
#' Reads every file in `paths`, numbering waves in the order given, and
#' binds the results so that columns missing in a wave are filled with `NA`.
#'
#' @param paths Character vector of paths to exported survey files.
#' @param format One of `"auto"`, `"csv"`, `"tsv"` or `"csv2"`.
#' @return A tibble with one row per respondent per wave.
#' @family wave readers
#' @export
read_all_waves <- function(paths, format = "auto") {
  if (length(paths) == 0) {
    stop("`paths` must contain at least one file.", call. = FALSE)
  }
  waves <- purrr::imap(unname(paths), function(path, i) {
    read_wave(path, wave = i, format = format)
  })
  dplyr::bind_rows(waves)
}

#' Standardise survey column names
#'
#' Lower-cases names, replaces runs of punctuation with underscores and
#' strips the question-number prefixes some survey tools add (`"Q12. "`).
#'
#' @param data A data frame of raw survey responses.
#' @return `data` with standardised column names.
#' @keywords internal
rename_survey_cols <- function(data) {
  new_names <- names(data) %>%
    stringr::str_remove("^[Qq][0-9]+[.:]\\s*") %>%
    stringr::str_to_lower() %>%
    stringr::str_replace_all("[^a-z0-9]+", "_") %>%
    stringr::str_remove_all("^_|_$")
  dupes <- duplicated(new_names)
  if (any(dupes)) {
    new_names <- make.unique(new_names, sep = "_")
  }
  names(data) <- new_names
  data
}

#' Check that a survey export exists
#'
#' @param path Path to the exported survey file.
#' @return `path`, invisibly, if the file exists; otherwise an error.
#' @noRd
check_wave_path <- function(path) {
  if (!is.character(path) || length(path) != 1) {
    stop("`path` must be a single string.", call. = FALSE)
  }
  if (!file.exists(path)) {
    stop("Survey export not found: ", path, call. = FALSE)
  }
  invisible(path)
}

#' Parse date and timestamp columns of a survey wave
#'
#' Survey platforms export start and end timestamps in different formats
#' depending on locale. This parses them, derives the interview duration
#' and flags interviews that are implausibly fast or end before they start.
#'
#' @param data A data frame of raw survey responses.
#' @param start_col,end_col Names of the start and end timestamp columns.
#' @param tz Time zone the timestamps were recorded in.
#' @param min_minutes Interviews shorter than this are flagged as speeders.
#' @return `data` with parsed timestamps and `duration_min`, `speeder` and
#'   `bad_times` columns.
#' @export
parse_wave_times <- function(data, start_col = "start_date", end_col = "end_date",
                             tz = "UTC", min_minutes = 3) {
  missing <- setdiff(c(start_col, end_col), names(data))
  if (length(missing) > 0) {
    stop("Timestamp columns not found: ", paste(missing, collapse = ", "),
      call. = FALSE)
  }
  orders <- c("Ymd HMS", "Ymd HM", "dmY HMS", "dmY HM", "mdY HMS", "mdY HM")
  parse_one <- function(x) {
    x <- stringr::str_squish(x)
    x[x == ""] <- NA_character_
    lubridate::parse_date_time(x, orders = orders, tz = tz, quiet = TRUE)
  }
  start <- parse_one(data[[start_col]])
  end <- parse_one(data[[end_col]])
  n_unparsed <- sum(!is.na(data[[start_col]]) & is.na(start)) +
    sum(!is.na(data[[end_col]]) & is.na(end))
  if (n_unparsed > 0) {
    warning(n_unparsed, " timestamp(s) could not be parsed and were set to NA.",
      call. = FALSE)
  }
  data[[start_col]] <- start
  data[[end_col]] <- end
  data %>%
    dplyr::mutate(
      duration_min = as.numeric(difftime(end, start, units = "mins")),
      bad_times = !is.na(duration_min) & duration_min < 0,
      speeder = !is.na(duration_min) & !bad_times & duration_min < min_minutes,
      interview_date = lubridate::as_date(start),
      interview_wday = lubridate::wday(start, label = TRUE, week_start = 1)
    )
}

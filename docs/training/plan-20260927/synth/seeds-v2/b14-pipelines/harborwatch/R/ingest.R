#' Standardise column names
#'
#' Lower-cases names, replaces runs of non-alphanumeric characters with a
#' single underscore and trims leading and trailing underscores.
#'
#' @param df A data frame with raw column names.
#' @return `df` with cleaned names.
#' @keywords internal
standardise_names <- function(df) {
  nms <- tolower(names(df))
  # "ETA (local)" -> "eta_local"
  nms <- gsub("[^a-z0-9]+", "_", nms)
  nms <- gsub("^_+|_+$", "", nms)

  # Two raw headers can collapse to the same name; keep both, suffixed
  names(df) <- make.unique(nms, sep = "_")
  df
}

#' Check that required columns are present
#'
#' @param df A data frame with standardised column names.
#' @param required A character vector of column names that must exist.
#' @param source A label for the input, used in the error message.
#' @return `df`, invisibly, if all columns are present.
#' @keywords internal
check_columns <- function(df, required, source) {
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop(
      sprintf("%s is missing column(s): %s", source, paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(df)
}

#' Read the vessel arrivals export
#'
#' Reads the harbour master's CSV export, standardises names and checks
#' that the columns used downstream are present.
#'
#' @param path Path to the input file.
#' @return A tibble with one row per arrival record.
#' @importFrom magrittr %>%
#' @keywords internal
read_arrivals_csv <- function(path) {
  # Everything as character: the export mixes date formats and placeholders,
  # and parsing happens in clean_arrivals()
  raw <- readr::read_csv(
    path,
    col_types = readr::cols(.default = readr::col_character()),
    na = c("", "NA", "-", "n/a"),
    progress = FALSE
  )

  arrivals <- standardise_names(raw)
  check_columns(
    arrivals,
    c("call_id", "vessel", "berth", "status", "eta", "ata", "atd"),
    basename(path)
  )

  arrivals %>%
    dplyr::mutate(source_file = basename(path))
}

#' Read the berth register
#'
#' Reads the berth register workbook, standardises names and checks that
#' the columns used downstream are present.
#'
#' @param path Path to the input file.
#' @param sheet The worksheet that holds the register.
#' @return A tibble with one row per berth.
#' @keywords internal
read_berths_xlsx <- function(path, sheet = "berths") {
  raw <- readxl::read_excel(path, sheet = sheet, .name_repair = "minimal")
  berths <- standardise_names(raw)
  check_columns(berths, c("berth", "terminal", "max_draught_m", "length_m"), basename(path))

  # Berth codes are typed by hand in the workbook; match the arrivals format
  berths %>%
    dplyr::mutate(
      berth = toupper(trimws(berth)),
      max_draught_m = as.numeric(max_draught_m),
      length_m = as.numeric(length_m)
    )
}

#' Read the tide predictions feed
#'
#' Reads the tide predictions JSON, standardises names and checks that the
#' columns used downstream are present.
#'
#' @param path Path to the input file.
#' @return A tibble with one row per predicted high or low water.
#' @keywords internal
read_tides_json <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyDataFrame = TRUE)$predictions
  tides <- standardise_names(tibble::as_tibble(raw))
  check_columns(tides, c("time", "type", "height_m"), basename(path))

  tides %>%
    dplyr::mutate(
      time = lubridate::ymd_hms(time, tz = "UTC"),
      height_m = as.numeric(height_m)
    )
}

#' Clean raw arrival records
#'
#' Parses time stamps, normalises berth codes, drops cancelled calls and
#' keeps only the latest record for each call when the export repeats it.
#'
#' @param arrivals A tibble returned by [read_arrivals_csv()].
#' @param tz The time zone the export uses for local times.
#' @return A cleaned tibble of arrivals.
#' @keywords internal
clean_arrivals <- function(arrivals, tz = "Europe/Berlin") {
  parse_time <- function(x) lubridate::ymd_hm(x, tz = tz, quiet = TRUE)

  arrivals %>%
    dplyr::mutate(
      berth = toupper(trimws(berth)),
      vessel = stringr::str_squish(vessel),
      eta = parse_time(eta),
      ata = parse_time(ata),
      atd = parse_time(atd)
    ) %>%
    dplyr::filter(!is.na(ata), is.na(status) | status != "CANCELLED") %>%
    # Export file names carry the export date, so the last file sorts first
    # and distinct() keeps its version of each call
    dplyr::arrange(call_id, dplyr::desc(source_file)) %>%
    dplyr::distinct(call_id, .keep_all = TRUE) %>%
    dplyr::mutate(arrival_date = as.Date(ata, tz = tz))
}

#' Attach berth details to arrivals
#'
#' Arrivals at berths that are not in the register are kept but reported
#' with a warning, since they usually mean the register is out of date.
#'
#' @param arrivals A cleaned tibble returned by [clean_arrivals()].
#' @param berths A tibble returned by [read_berths_xlsx()].
#' @return `arrivals` with terminal and berth size columns added.
#' @keywords internal
join_berths <- function(arrivals, berths) {
  unknown <- setdiff(unique(arrivals$berth), berths$berth)

  if (length(unknown) > 0) {
    warning(
      "Berths not in the register: ", paste(sort(unknown), collapse = ", "),
      call. = FALSE
    )
  }

  arrivals %>%
    dplyr::left_join(
      berths %>% dplyr::select(berth, terminal, max_draught_m, length_m),
      by = "berth"
    ) %>%
    dplyr::mutate(terminal = dplyr::coalesce(terminal, "UNKNOWN"))
}

#' Check that required columns are present
#'
#' @param data A data frame of survey responses.
#' @param cols Character vector of column names that must exist.
#' @return A one-row tibble describing the check result.
#' @export
check_required_cols <- function(data, cols) {
  missing <- setdiff(cols, names(data))
  new_check_result(
    check = "required_cols",
    passed = length(missing) == 0,
    n_failed = length(missing),
    detail = paste(missing, collapse = ", ")
  )
}

#' Check that a column has no duplicated values
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @return A one-row tibble describing the check result.
#' @export
check_unique <- function(data, col) {
  dupes <- data %>%
    dplyr::count({{ col }}, name = "n_rows") %>%
    dplyr::filter(.data$n_rows > 1)
  new_check_result(
    check = paste0("unique_", rlang::as_label(rlang::enquo(col))),
    passed = nrow(dupes) == 0,
    n_failed = sum(dupes$n_rows),
    detail = paste(utils::head(dupes[[1]], 5), collapse = ", ")
  )
}

#' Check that a numeric column lies within a range
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @param lower,upper Inclusive bounds of the allowed range.
#' @return A one-row tibble describing the check result.
#' @export
check_range <- function(data, col, lower, upper) {
  values <- dplyr::pull(data, {{ col }})

  # Missing answers are the business of check_missing_rate(), not this one.
  bad <- !is.na(values) & (values < lower | values > upper)
  new_check_result(
    check = paste0("range_", rlang::as_label(rlang::enquo(col))),
    passed = !any(bad),
    n_failed = sum(bad),
    detail = paste(utils::head(unique(values[bad]), 5), collapse = ", ")
  )
}

#' Check that a column only contains allowed values
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @param allowed Vector of permitted values; `NA` is always permitted.
#' @return A one-row tibble describing the check result.
#' @export
check_allowed <- function(data, col, allowed) {
  values <- dplyr::pull(data, {{ col }})
  bad <- !is.na(values) & !(values %in% allowed)
  new_check_result(
    check = paste0("allowed_", rlang::as_label(rlang::enquo(col))),
    passed = !any(bad),
    n_failed = sum(bad),
    detail = paste(utils::head(unique(values[bad]), 5), collapse = ", ")
  )
}

#' Check the share of missing answers in a column
#'
#' @param data A data frame of survey responses.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> The column to check.
#' @param max_prop Largest acceptable share of missing values.
#' @return A one-row tibble describing the check result.
#' @export
check_missing_rate <- function(data, col, max_prop = 0.2) {
  values <- dplyr::pull(data, {{ col }})
  prop <- if (length(values) == 0) 0 else mean(is.na(values))
  new_check_result(
    check = paste0("missing_", rlang::as_label(rlang::enquo(col))),
    passed = prop <= max_prop,
    n_failed = sum(is.na(values)),
    detail = sprintf("%.1f%% missing", 100 * prop)
  )
}

#' Run a list of checks and combine the results
#'
#' Each element of `checks` is a function taking the data and returning a
#' check result; errors inside a check are captured as failed rows rather
#' than aborting the whole run.
#'
#' @param data A data frame of survey responses.
#' @param checks A named list of functions, each returning a check result.
#' @param stop_on_fail If `TRUE`, raise an error when any check fails.
#' @return A tibble with one row per check.
#' @export
run_checks <- function(data, checks, stop_on_fail = FALSE) {
  # A check that warns is treated as failed: warnings here usually mean a
  # type coercion went wrong and the comparison cannot be trusted.
  results <- purrr::imap(checks, function(fn, nm) {
    tryCatch(
      fn(data),
      error = function(e) {
        new_check_result(nm, passed = FALSE, n_failed = NA_integer_,
          detail = conditionMessage(e))
      },
      warning = function(w) {
        new_check_result(nm, passed = FALSE, n_failed = NA_integer_,
          detail = conditionMessage(w))
      }
    )
  })

  report <- dplyr::bind_rows(results)
  if (stop_on_fail && !all(report$passed)) {
    failed <- report$check[!report$passed]
    stop("Failed checks: ", paste(failed, collapse = ", "), call. = FALSE)
  }
  report
}

#' Build a single check result row
#'
#' @param check Name of the check.
#' @param passed Whether the check passed.
#' @param n_failed Number of offending rows or columns.
#' @param detail Short human-readable detail string.
#' @noRd
new_check_result <- function(check, passed, n_failed, detail = "") {
  tibble::tibble(
    check = check,
    passed = isTRUE(passed),
    n_failed = as.integer(n_failed),
    detail = if (isTRUE(passed)) "" else detail
  )
}

#' Validate a cleaned survey wave against its codebook
#'
#' Builds the standard set of checks from a codebook (one row per item,
#' with `item`, `type`, `min`, `max` and `levels` columns), runs them and
#' optionally writes the report to disk.
#'
#' @param data A data frame of survey responses.
#' @param codebook A data frame describing each item.
#' @param id_col Name of the respondent identifier column.
#' @param report_path Optional path to write the report as CSV.
#' @return A tibble with one row per check, invisibly if `report_path` is set.
#' @export
validate_wave <- function(data, codebook, id_col = "respondent_id", report_path = NULL) {
  checks <- list(
    required_cols = function(d) check_required_cols(d, c(id_col, codebook$item)),
    unique_id = function(d) check_unique(d, .data[[id_col]])
  )

  # Every per-item check is built inside local() so that each closure keeps
  # its own copy of the item name and bounds instead of the loop's last one.
  for (i in seq_len(nrow(codebook))) {
    item <- codebook$item[[i]]
    type <- codebook$type[[i]]
    if (!item %in% names(data)) {
      next
    }

    if (identical(type, "numeric")) {
      checks[[paste0("range_", item)]] <- local({
        it <- item
        lower <- codebook$min[[i]]
        upper <- codebook$max[[i]]
        function(d) check_range(d, .data[[it]], lower, upper)
      })
    } else if (identical(type, "categorical")) {
      checks[[paste0("allowed_", item)]] <- local({
        it <- item
        allowed <- stringr::str_split_1(codebook$levels[[i]], "\\s*\\|\\s*")
        function(d) check_allowed(d, .data[[it]], allowed)
      })
    }
    checks[[paste0("missing_", item)]] <- local({
      it <- item
      function(d) check_missing_rate(d, .data[[it]])
    })
  }

  report <- run_checks(data, checks)
  n_failed <- sum(!report$passed)
  message(sprintf("%d of %d checks passed.", nrow(report) - n_failed, nrow(report)))

  if (!is.null(report_path)) {
    readr::write_csv(report, report_path)
    return(invisible(report))
  }
  report
}

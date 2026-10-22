#' Add gross margin columns
#'
#' @param sales A data frame of till transactions with `revenue`,
#'   `unitCost` and `units` columns.
#' @return `sales` with `grossProfit` and `grossMargin` columns.
#' @family margin helpers
#' @export
addGrossMargin <- function(sales) {
  sales |>
    dplyr::mutate(
      grossProfit = .data$revenue - .data$unitCost * .data$units,
      grossMargin = dplyr::if_else(.data$revenue > 0, .data$grossProfit / .data$revenue,
        NA_real_)
    )
}

#' Add markdown rate columns
#'
#' @param sales A data frame of till transactions with `revenue`,
#'   `listPrice` and `units` columns.
#' @return `sales` with `markdownAmount` and `markdownRate` columns.
#' @family margin helpers
#' @export
addMarkdownRate <- function(sales) {
  sales |>
    dplyr::mutate(
      markdownAmount = .data$listPrice * .data$units - .data$revenue,
      markdownRate = dplyr::if_else(.data$listPrice > 0,
        .data$markdownAmount / (.data$listPrice * .data$units), NA_real_)
    )
}

#' Add return rate columns
#'
#' @param sales A data frame of till transactions with `units` and
#'   `returnedUnits` columns.
#' @return `sales` with `netUnits` and `returnRate` columns.
#' @family margin helpers
#' @export
addReturnRate <- function(sales) {
  sales |>
    dplyr::mutate(
      netUnits = .data$units - .data$returnedUnits,
      returnRate = dplyr::if_else(.data$units > 0, .data$returnedUnits / .data$units,
        NA_real_)
    )
}

#' Flag products whose margin falls below a threshold
#'
#' @param sales A data frame of till transactions with `grossMargin`.
#' @param threshold Margin below which a row is flagged.
#' @param col <[`data-masking`][dplyr::dplyr_data_masking]> Name of the new flag column.
#' @return `sales` with an added logical flag column.
#' @export
flagLowMargin <- function(sales, threshold = 0.15, col = lowMargin) {
  if (!"grossMargin" %in% names(sales)) {
    sales <- addGrossMargin(sales)
  }
  sales |>
    dplyr::mutate({{ col }} := !is.na(.data$grossMargin) & .data$grossMargin < threshold)
}

#' Tidy free-text product names from the till system
#'
#' Removes size suffixes, collapses whitespace and converts names to title
#' case so that the same product is not split across spellings.
#'
#' @param names A character vector of raw product names.
#' @return A character vector of cleaned names.
#' @keywords internal
cleanProductNames <- function(names) {
  names |>
    stringr::str_replace_all("_", " ") |>
    stringr::str_remove("\\s*\\(?\\d+(\\.\\d+)?\\s*(g|kg|ml|l|pk)\\)?$") |>
    stringr::str_squish() |>
    stringr::str_to_title()
}

#' Split SKU codes into their components
#'
#' SKUs follow the pattern `DEPT-CLASS-ITEM`, e.g. `"GRO-0412-00087"`.
#' Malformed codes produce `NA` components rather than an error.
#'
#' @param sales A data frame of till transactions.
#' @param skuCol <[`tidy-select`][dplyr::dplyr_tidy_select]> Column holding the SKU.
#' @return `sales` with `department`, `classCode` and `itemCode` columns.
#' @export
splitSku <- function(sales, skuCol = sku) {
  sales |>
    tidyr::separate_wider_regex(
      {{ skuCol }},
      patterns = c(department = "[A-Z]{3}", "-", classCode = "\\d{4}", "-",
        itemCode = "\\d{5}"),
      too_few = "align_start",
      cols_remove = FALSE
    ) |>
    dplyr::mutate(department = forcats::fct_infreq(.data$department))
}

#' Summarise margins per category
#'
#' @param sales A data frame of till transactions with `revenue`,
#'   `unitCost` and `units` columns.
#' @param categoryCol <[`data-masking`][dplyr::dplyr_data_masking]> Column holding the category.
#' @return A tibble with one row per category, ordered by margin.
#' @export
marginByCategory <- function(sales, categoryCol = category) {
  sales |>
    addGrossMargin() |>
    dplyr::group_by({{ categoryCol }}) |>
    dplyr::summarise(
      revenue = sum(.data$revenue, na.rm = TRUE),
      grossProfit = sum(.data$grossProfit, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      grossMargin = .data$grossProfit / .data$revenue,
      {{ categoryCol }} := forcats::fct_reorder(
        as.character({{ categoryCol }}), .data$grossMargin, .desc = TRUE
      )
    ) |>
    dplyr::arrange({{ categoryCol }})
}

#' Summarise margins per store
#'
#' @param sales A data frame of till transactions with `revenue`,
#'   `unitCost` and `units` columns.
#' @param storeCol <[`data-masking`][dplyr::dplyr_data_masking]> Column identifying the store.
#' @return A tibble with one row per store, ordered by margin.
#' @export
marginByStore <- function(sales, storeCol = storeId) {
  sales |>
    addGrossMargin() |>
    dplyr::group_by({{ storeCol }}) |>
    dplyr::summarise(
      revenue = sum(.data$revenue, na.rm = TRUE),
      grossProfit = sum(.data$grossProfit, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      grossMargin = .data$grossProfit / .data$revenue,
      {{ storeCol }} := forcats::fct_reorder(
        as.character({{ storeCol }}), .data$grossMargin, .desc = TRUE
      )
    ) |>
    dplyr::arrange({{ storeCol }})
}

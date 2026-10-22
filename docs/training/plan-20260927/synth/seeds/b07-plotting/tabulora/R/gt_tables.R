#' Report table with house styling
#'
#' Builds a [gt::gt()] table with a title, optional subtitle and the house
#' header and body styles applied.
#'
#' @param data A data frame to be shown as a table.
#' @param title Table title.
#' @param subtitle Optional table subtitle.
#' @param rowname_col Optional name of the column used as row labels.
#' @return A `gt_tbl` object.
#' @family gt tables
#' @importFrom gt gt tab_header tab_options md
#' @importFrom magrittr %>%
#' @export
gt_report <- function(data, title, subtitle = NULL, rowname_col = NULL) {
  tbl <- gt(data, rowname_col = rowname_col) %>%
    tab_header(
      title = md(paste0("**", title, "**")),
      subtitle = if (is.null(subtitle)) NULL else md(subtitle)
    )
  tbl %>%
    gt_style_house() %>%
    tab_options(table.width = gt::pct(100))
}

#' Apply house header and body styles
#'
#' Sets fonts, border colours and row striping used in all tabulora gt
#' tables.
#'
#' @param tbl A `gt_tbl` object.
#' @param stripe Whether to stripe alternate body rows.
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt tab_options opt_row_striping
#' @export
gt_style_house <- function(tbl, stripe = TRUE) {
  tbl <- tbl %>%
    tab_options(
      heading.align = "left",
      heading.border.bottom.color = "#1f4e79",
      column_labels.font.weight = "bold",
      column_labels.border.bottom.color = "#1f4e79",
      column_labels.border.bottom.width = gt::px(2),
      table.border.top.color = "transparent",
      table.font.size = gt::px(13),
      data_row.padding = gt::px(4)
    )
  if (stripe) {
    tbl <- tbl %>% opt_row_striping()
  }
  tbl
}

#' Highlight rows that meet a condition
#'
#' Fills the background of every body row for which `rows` is `TRUE` and
#' optionally makes the text bold.
#'
#' @param tbl A `gt_tbl` object.
#' @param rows A logical vector, one value per body row.
#' @param fill Background colour of highlighted rows.
#' @param bold Whether to make the highlighted text bold.
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt tab_style cell_fill cell_text cells_body
#' @export
gt_highlight_rows <- function(tbl, rows, fill = "#fff3cd", bold = FALSE) {
  idx <- which(rows)
  if (length(idx) == 0) {
    return(tbl)
  }
  styles <- list(cell_fill(color = fill))
  if (bold) {
    styles <- c(styles, list(cell_text(weight = "bold")))
  }
  tbl %>% tab_style(style = styles, locations = cells_body(rows = idx))
}

#' Colour a numeric column by value
#'
#' Applies a colour scale to one numeric column, using a diverging palette
#' when the values span zero and a sequential one otherwise.
#'
#' @param tbl A `gt_tbl` object.
#' @param column Name of the numeric column to colour.
#' @param palette Optional character vector of colours to use instead.
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt data_color
#' @export
gt_colour_column <- function(tbl, column, palette = NULL) {
  values <- tbl[["_data"]][[column]]
  if (!is.numeric(values)) {
    stop("Column `", column, "` is not numeric.", call. = FALSE)
  }
  rng <- range(values, na.rm = TRUE)
  if (is.null(palette)) {
    palette <- if (rng[1] < 0 && rng[2] > 0) {
      paletteDiverging(7)
    } else {
      rev(paletteGet("slate"))
    }
  }
  # Centre the diverging scale on zero so that colour intensity is symmetric
  if (rng[1] < 0 && rng[2] > 0) {
    rng <- c(-1, 1) * max(abs(rng))
  }
  tbl %>% data_color(columns = column, palette = palette, domain = rng)
}

#' Format columns according to their type
#'
#' Applies number formatting to doubles, integer formatting to integers,
#' and date formatting to dates, so callers do not have to list columns.
#'
#' @param tbl A `gt_tbl` object.
#' @param decimals Number of decimal places for double columns.
#' @param date_style Date style passed to [gt::fmt_date()].
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt fmt_number fmt_integer fmt_date
#' @export
gt_format_by_type <- function(tbl, decimals = 2, date_style = "iso") {
  data <- tbl[["_data"]]
  dbl_cols <- names(data)[vapply(data, is.double, logical(1)) &
                            !vapply(data, inherits, logical(1), what = "Date")]
  int_cols <- names(data)[vapply(data, is.integer, logical(1))]
  date_cols <- names(data)[vapply(data, inherits, logical(1), what = "Date")]
  if (length(dbl_cols) > 0) {
    tbl <- tbl %>% fmt_number(columns = dplyr::all_of(dbl_cols), decimals = decimals)
  }
  if (length(int_cols) > 0) {
    tbl <- tbl %>% fmt_integer(columns = dplyr::all_of(int_cols))
  }
  if (length(date_cols) > 0) {
    tbl <- tbl %>% fmt_date(columns = dplyr::all_of(date_cols), date_style = date_style)
  }
  tbl
}

#' Add footnotes to column labels
#'
#' Attaches one footnote per named element of `notes` to the matching
#' column label.
#'
#' @param tbl A `gt_tbl` object.
#' @param notes A named character vector mapping column names to footnotes.
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt tab_footnote cells_column_labels
#' @export
gt_add_column_notes <- function(tbl, notes) {
  if (is.null(names(notes)) || any(!nzchar(names(notes)))) {
    stop("`notes` must be a named character vector.", call. = FALSE)
  }
  for (col in names(notes)) {
    tbl <- tbl %>%
      tab_footnote(
        footnote = notes[[col]],
        locations = cells_column_labels(columns = dplyr::all_of(col))
      )
  }
  tbl
}

#' Add a source note
#'
#' Appends a `"Source: ..."` line under the table, in a smaller grey font.
#'
#' @param tbl A `gt_tbl` object.
#' @param source A single string naming the data source.
#' @return The modified `gt_tbl` object.
#' @family gt styling
#' @importFrom gt tab_source_note tab_options
#' @export
gt_add_source <- function(tbl, source) {
  tbl %>%
    tab_source_note(source_note = paste0("Source: ", source)) %>%
    tab_options(source_notes.font.size = gt::px(11))
}

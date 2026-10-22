#' Label the axes and titles of a plot
#'
#' Sets the axis titles, title, subtitle and caption in one call, leaving
#' any argument that is `NULL` untouched.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param x Title of the x axis.
#' @param y Title of the y axis.
#' @param title Plot title.
#' @param subtitle Plot subtitle.
#' @param caption Plot caption.
#' @return The modified plot.
#' @export
label_plot <- function(plot, x = NULL, y = NULL, title = NULL, subtitle = NULL,
                       caption = NULL) {
  args <- list(x = x, y = y, title = title, subtitle = subtitle, caption = caption)

  # NULL means "leave as is", so drop those before calling labs()
  args <- Filter(Negate(is.null), args)
  if (length(args) == 0) {
    return(plot)
  }

  plot + do.call(ggplot2::labs, args)
}

#' Turn a column name into a readable label
#'
#' Replaces underscores and dots with spaces and capitalises the first
#' letter, so `"mean_temp_c"` becomes `"Mean temp c"`.
#'
#' @param x Character vector of column names.
#' @return A character vector the same length as `x`.
#' @export
humanise_name <- function(x) {
  x <- gsub("[._]+", " ", x)
  x <- trimws(x)
  out <- paste0(toupper(substring(x, 1, 1)), substring(x, 2))
  out[is.na(x)] <- NA_character_
  out
}

#' Append a unit to a label
#'
#' @param x Character vector of labels.
#' @param unit A single unit string, for example `"kg"` or `"mm"`.
#' @param style Whether to wrap the unit in parentheses or brackets.
#' @return A character vector the same length as `x`.
#' @export
label_with_unit <- function(x, unit, style = c("parens", "brackets")) {
  style <- match.arg(style)
  switch(style,
    parens = paste0(x, " (", unit, ")"),
    brackets = paste0(x, " [", unit, "]")
  )
}

#' Format p-values for display
#'
#' Rounds p-values to a fixed number of digits and replaces values below the
#' display threshold with a `"< 0.001"` style string.
#'
#' @param p Numeric vector of p-values.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `p`.
#' @export
format_p_value <- function(p, digits = 3) {
  threshold <- 10^(-digits)
  out <- formatC(p, format = "f", digits = digits)

  # Anything that would round to zero is shown as "< threshold"
  out[!is.na(p) & p < threshold] <- paste0("< ", formatC(threshold, format = "f", digits = digits))
  out[is.na(p)] <- ""
  out
}

#' Format an estimate with its confidence interval
#'
#' Produces strings of the form `"1.23 (0.98, 1.51)"`.
#'
#' @param estimate Numeric vector of point estimates.
#' @param lower Numeric vector of lower interval bounds.
#' @param upper Numeric vector of upper interval bounds.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `estimate`.
#' @export
format_estimate_ci <- function(estimate, lower, upper, digits = 2) {
  fmt <- function(v) formatC(v, format = "f", digits = digits)
  out <- paste0(fmt(estimate), " (", fmt(lower), ", ", fmt(upper), ")")
  out[is.na(estimate)] <- ""
  out
}

#' Format a relative change between two values
#'
#' Produces signed percentage strings such as `"+4.2%"` or `"-10.0%"`.
#'
#' @param old Numeric vector of reference values.
#' @param new Numeric vector of comparison values.
#' @param digits Number of decimal places.
#' @return A character vector the same length as `old`.
#' @export
format_percent_change <- function(old, new, digits = 1) {
  # abs() keeps the sign meaningful when the reference value is negative
  change <- 100 * (new - old) / abs(old)
  sign <- ifelse(change > 0, "+", "")
  out <- paste0(sign, formatC(change, format = "f", digits = digits), "%")
  out[is.na(change) | !is.finite(change)] <- ""
  out
}

#' Add a data source line to the caption
#'
#' Appends `"Source: ..."` and, optionally, the date the data were
#' retrieved to any existing caption.
#'
#' @param plot A [ggplot2::ggplot()] object.
#' @param source A single string naming the data source.
#' @param date Optional retrieval date; anything [format()] accepts.
#' @return The modified plot.
#' @export
add_caption_source <- function(plot, source, date = NULL) {
  line <- paste0("Source: ", source)

  if (!is.null(date)) {
    line <- paste0(line, " (retrieved ", format(as.Date(date), "%d %b %Y"), ")")
  }

  # Keep any caption that is already there and add the source below it
  existing <- plot$labels$caption
  if (!is.null(existing) && nzchar(existing)) {
    line <- paste(existing, line, sep = "\n")
  }

  plot + ggplot2::labs(caption = line)
}

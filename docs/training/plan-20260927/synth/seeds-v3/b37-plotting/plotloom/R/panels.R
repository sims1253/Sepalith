#' Signal a panel type mismatch
#'
#' @param name Variable name used for the panel title and legend.
#' @param kind Panel kind that was requested.
#' @return Never returns; always signals an error.
#' @keywords internal
panel_type_error <- function(name, kind) {
  stop(sprintf("`%s` cannot be drawn as a %s panel.", name, kind), call. = FALSE)
}

#' Draw a numeric panel
#'
#' Draws a histogram of the finite values of `x` into the current figure
#' region, with axes, a title and a legend giving the number of values.
#'
#' @param x A numeric vector.
#' @param name Variable name used for the panel title and legend.
#' @param col Fill colour for the panel marks.
#' @param breaks Number of histogram bins.
#' @return Invisibly, a list with elements `name`, `kind`, `n` and
#'   `n_missing`.
#' @family panels
#' @export
panel_numeric <- function(x, name, col = "grey70", breaks = 20L) {
  if (!is.numeric(x)) {
    panel_type_error(name, "numeric")
  }
  ok <- x[is.finite(x)]
  h <- hist(ok, breaks = breaks, plot = FALSE)
  n_bins <- length(h$counts)

  old <- par(mar = c(4, 4, 2, 1))
  on.exit(par(old), add = TRUE)
  plot.new()
  plot.window(xlim = range(h$breaks), ylim = c(0, max(h$counts)))
  rect(h$breaks[-(n_bins + 1L)], 0, h$breaks[-1L], h$counts, col = col)
  axis(1)
  axis(2, las = 1)
  title(main = name, xlab = name, ylab = "count")
  box()
  legend("topright", legend = sprintf("n = %d", length(ok)), bty = "n")
  invisible(list(
    name = name,
    kind = "numeric",
    n = length(ok),
    n_missing = length(x) - length(ok)
  ))
}

#' Draw a factor panel
#'
#' Draws a bar chart of the most frequent levels of `x` into the current
#' figure region, with axes, a title and a legend giving the number of values.
#'
#' @param x A factor or character vector.
#' @param name Variable name used for the panel title and legend.
#' @param col Fill colour for the panel marks.
#' @param max_levels Largest number of levels to show.
#' @return Invisibly, a list with elements `name`, `kind`, `n` and
#'   `n_missing`.
#' @family panels
#' @export
panel_factor <- function(x, name, col = "grey70", max_levels = 12L) {
  if (!is.factor(x) && !is.character(x)) {
    panel_type_error(name, "factor")
  }
  ok <- x[!is.na(x)]
  counts <- sort(table(ok), decreasing = TRUE)
  counts <- counts[seq_len(min(length(counts), max_levels))]
  at <- seq_along(counts)

  old <- par(mar = c(4, 4, 2, 1))
  on.exit(par(old), add = TRUE)
  plot.new()
  plot.window(xlim = c(0.5, length(counts) + 0.5), ylim = c(0, max(counts)))
  rect(at - 0.4, 0, at + 0.4, counts, col = col)
  axis(1, at = at, labels = names(counts))
  axis(2, las = 1)
  title(main = name, xlab = name, ylab = "count")
  box()
  legend("topright", legend = sprintf("n = %d", length(ok)), bty = "n")
  invisible(list(
    name = name,
    kind = "factor",
    n = length(ok),
    n_missing = length(x) - length(ok)
  ))
}

#' Draw a logical panel
#'
#' Draws a two-bar chart of the `FALSE` and `TRUE` counts of `x` into the
#' current figure region, with axes, a title and a legend giving the number
#' of values.
#'
#' @param x A logical vector.
#' @param name Variable name used for the panel title and legend.
#' @param col Fill colour for the panel marks.
#' @return Invisibly, a list with elements `name`, `kind`, `n` and
#'   `n_missing`.
#' @family panels
#' @export
panel_logical <- function(x, name, col = "grey70") {
  if (!is.logical(x)) {
    panel_type_error(name, "logical")
  }
  ok <- x[!is.na(x)]
  counts <- c(`FALSE` = sum(!ok), `TRUE` = sum(ok))
  at <- seq_along(counts)

  old <- par(mar = c(4, 4, 2, 1))
  on.exit(par(old), add = TRUE)
  plot.new()
  plot.window(xlim = c(0.5, length(counts) + 0.5), ylim = c(0, max(counts, 1L)))
  rect(at - 0.4, 0, at + 0.4, counts, col = col)
  axis(1, at = at, labels = names(counts))
  axis(2, las = 1)
  title(main = name, xlab = name, ylab = "count")
  box()
  legend("topright", legend = sprintf("n = %d", length(ok)), bty = "n")
  invisible(list(
    name = name,
    kind = "logical",
    n = length(ok),
    n_missing = length(x) - length(ok)
  ))
}

#' Draw a date panel
#'
#' Draws a histogram of the non-missing dates in `x` into the current figure
#' region, with date labels on the x axis, a title and a legend giving the
#' number of values.
#'
#' @param x A `Date` vector.
#' @param name Variable name used for the panel title and legend.
#' @param col Fill colour for the panel marks.
#' @param breaks Number of histogram bins.
#' @return Invisibly, a list with elements `name`, `kind`, `n` and
#'   `n_missing`.
#' @family panels
#' @export
panel_date <- function(x, name, col = "grey70", breaks = 20L) {
  if (!inherits(x, "Date")) {
    panel_type_error(name, "date")
  }
  ok <- x[!is.na(x)]
  h <- hist(as.numeric(ok), breaks = breaks, plot = FALSE)
  n_bins <- length(h$counts)

  old <- par(mar = c(4, 4, 2, 1))
  on.exit(par(old), add = TRUE)
  plot.new()
  plot.window(xlim = range(h$breaks), ylim = c(0, max(h$counts)))
  rect(h$breaks[-(n_bins + 1L)], 0, h$breaks[-1L], h$counts, col = col)
  ticks <- pretty(h$breaks)
  axis(1, at = ticks, labels = format(as.Date(ticks, origin = "1970-01-01")))
  axis(2, las = 1)
  title(main = name, xlab = name, ylab = "count")
  box()
  legend("topright", legend = sprintf("n = %d", length(ok)), bty = "n")
  invisible(list(
    name = name,
    kind = "date",
    n = length(ok),
    n_missing = length(x) - length(ok)
  ))
}

#' Guess the panel kind of a variable
#'
#' @param x A vector.
#' @return One of `"numeric"`, `"factor"`, `"logical"` or `"date"`, or
#'   `NA_character_` when no panel fits.
#' @keywords internal
panel_kind <- function(x) {
  if (inherits(x, "Date")) {
    return("date")
  }
  if (is.logical(x)) {
    return("logical")
  }
  if (is.numeric(x)) {
    return("numeric")
  }
  if (is.factor(x) || is.character(x)) {
    return("factor")
  }
  NA_character_
}

#' Draw one panel per variable
#'
#' Splits the device into a grid with `ncol` columns and draws one panel for
#' each of `vars`, choosing the panel function from the column type.
#'
#' @param data A data frame.
#' @param vars Names of the columns to draw; defaults to all columns.
#' @param ncol Number of panel columns.
#' @param col Fill colour for the panel marks.
#' @return Invisibly, a data frame with columns `name`, `kind`, `n` and
#'   `n_missing`, one row per drawn panel.
#' @export
draw_panels <- function(data, vars = names(data), ncol = 2L, col = "grey70") {
  missing_vars <- setdiff(vars, names(data))
  if (length(missing_vars) > 0L) {
    stop(sprintf("Unknown variable(s): %s.", paste(missing_vars, collapse = ", ")),
         call. = FALSE)
  }
  nrow <- ceiling(length(vars) / ncol)
  old <- par(mfrow = c(nrow, ncol))
  on.exit(par(old), add = TRUE)

  out <- vector("list", length(vars))
  for (i in seq_along(vars)) {
    name <- vars[[i]]
    x <- data[[name]]
    kind <- panel_kind(x)
    if (is.na(kind)) {
      warning(sprintf("Skipping `%s`: no panel for class %s.", name, class(x)[1]),
              call. = FALSE)
      plot.new()
      next
    }
    out[[i]] <- switch(kind,
      numeric = panel_numeric(x, name, col = col),
      factor = panel_factor(x, name, col = col),
      logical = panel_logical(x, name, col = col),
      date = panel_date(x, name, col = col)
    )
  }
  out <- Filter(Negate(is.null), out)
  invisible(do.call(rbind, lapply(out, as.data.frame)))
}

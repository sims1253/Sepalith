#' Named margin presets
#'
#' Each preset is derived from `base` by scaling or zeroing sides, so a
#' custom `base` shifts every preset consistently.
#'
#' @param style Preset name.
#' @param base Named numeric vector of default margins in lines, with
#'   elements `bottom`, `left`, `top` and `right`.
#' @return A numeric vector of length four suitable for `par(mar = )`.
#' @export
margin_preset <- function(style = c("default", "compact", "wide", "title", "none"),
                          base = c(bottom = 5.1, left = 4.1, top = 4.1, right = 2.1)) {
  style <- match.arg(style)
  sides <- c("bottom", "left", "top", "right")
  if (!is.numeric(base) || !identical(names(base), sides)) {
    stop("`base` must be a numeric vector named bottom, left, top, right.", call. = FALSE)
  }
  mar <- switch(style,
    default = {
      base
    },
    compact = {
      out <- base / 2
      out[["top"]] <- base[["top"]] / 4
      out
    },
    wide = {
      out <- base
      out[["left"]] <- base[["left"]] * 2
      out[["right"]] <- base[["right"]] * 2
      out
    },
    title = {
      out <- base / 2
      out[["top"]] <- base[["top"]]
      out
    },
    none = {
      base * 0
    }
  )
  unname(mar)
}

#' Choose rows and columns for a panel grid
#'
#' Fills in whichever of `nrow` and `ncol` is missing so that the grid holds
#' at least `n` panels; when both are missing the grid is as square as
#' possible with more columns than rows.
#'
#' @param n Number of panels.
#' @param nrow,ncol Fixed number of rows or columns, or `NULL`.
#' @return An integer vector `c(nrow, ncol)`.
#' @export
panel_grid <- function(n, nrow = NULL, ncol = NULL) {
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop("`n` must be a single positive number.", call. = FALSE)
  }
  if (!is.null(nrow) && !is.null(ncol)) {
    if (nrow * ncol < n) {
      stop(sprintf("A %d x %d grid cannot hold %d panels.", nrow, ncol, n), call. = FALSE)
    }
    return(as.integer(c(nrow, ncol)))
  }
  if (is.null(nrow) && is.null(ncol)) {
    ncol <- ceiling(sqrt(n))
    nrow <- ceiling(n / ncol)
    return(as.integer(c(nrow, ncol)))
  }

  # one side fixed: derive the other
  if (is.null(nrow)) {
    if (ncol < 1) {
      stop("`ncol` must be at least 1.", call. = FALSE)
    }
    nrow <- ceiling(n / ncol)
  } else {
    if (nrow < 1) {
      stop("`nrow` must be at least 1.", call. = FALSE)
    }
    ncol <- ceiling(n / nrow)
  }
  as.integer(c(nrow, ncol))
}

#' Build a layout with a legend strip
#'
#' @description
#' Builds the arguments for [graphics::layout()] for `n` panels in a grid
#' plus one legend cell spanning the `legend` side. The legend cell is
#' always the last figure number, so it is drawn after all panels.
#'
#' @param n Number of panels.
#' @param ncol Number of panel columns.
#' @param legend Side of the legend strip.
#' @param size Relative size of the legend strip compared with one panel.
#' @return A list with elements `mat`, `widths` and `heights`.
#' @export
layout_with_legend <- function(n, ncol = 2L, legend = c("none", "right", "left", "bottom", "top"),
                               size = 0.3) {
  legend <- match.arg(legend)
  dims <- panel_grid(n, ncol = ncol)
  nrow <- dims[[1]]
  ncol <- dims[[2]]
  ids <- seq_len(nrow * ncol)
  ids[ids > n] <- 0L
  mat <- matrix(ids, nrow = nrow, ncol = ncol, byrow = TRUE)
  widths <- rep(1, ncol)
  heights <- rep(1, nrow)
  key <- n + 1L

  switch(legend,
    none = NULL,
    right = {
      mat <- cbind(mat, key)
      widths <- c(widths, size)
    },
    left = {
      mat <- cbind(key, mat)
      widths <- c(size, widths)
    },
    bottom = {
      mat <- rbind(mat, key)
      heights <- c(heights, size)
    },
    top = {
      mat <- rbind(key, mat)
      heights <- c(size, heights)
    }
  )
  dimnames(mat) <- NULL
  list(mat = mat, widths = widths, heights = heights)
}

#' Validate graphical parameter settings
#'
#' Checks the subset of `par()` settings that plotloom functions accept.
#' Only the fields present in `settings` are checked.
#'
#' @param settings A named list with any of `mar`, `oma`, `mgp`, `mfrow`,
#'   `cex` and `las`.
#' @return `settings`, invisibly, if all present fields are valid.
#' @keywords internal
check_par_settings <- function(settings) {
  known <- c("mar", "oma", "mgp", "mfrow", "cex", "las")
  unknown <- setdiff(names(settings), known)
  if (length(unknown) > 0L) {
    stop(sprintf("Unsupported par setting(s): %s.", paste(unknown, collapse = ", ")),
         call. = FALSE)
  }

  # Validate `mar`
  if (!is.null(settings$mar)) {
    if (!is.numeric(settings$mar) || length(settings$mar) != 4L || any(settings$mar < 0)) {
      stop("`mar` must be 4 non-negative numbers.", call. = FALSE)
    }
  }
  # Validate `oma`
  if (!is.null(settings$oma)) {
    if (!is.numeric(settings$oma) || length(settings$oma) != 4L || any(settings$oma < 0)) {
      stop("`oma` must be 4 non-negative numbers.", call. = FALSE)
    }
  }
  # Validate `mgp`
  if (!is.null(settings$mgp)) {
    if (!is.numeric(settings$mgp) || length(settings$mgp) != 3L || any(settings$mgp < 0)) {
      stop("`mgp` must be 3 non-negative numbers.", call. = FALSE)
    }
  }
  # Validate `mfrow`
  if (!is.null(settings$mfrow)) {
    if (!is.numeric(settings$mfrow) || length(settings$mfrow) != 2L || any(settings$mfrow < 1)) {
      stop("`mfrow` must be 2 positive numbers.", call. = FALSE)
    }
  }
  # Validate `cex`
  if (!is.null(settings$cex)) {
    if (!is.numeric(settings$cex) || length(settings$cex) != 1L || settings$cex <= 0) {
      stop("`cex` must be 1 positive number.", call. = FALSE)
    }
  }
  # Validate `las`
  if (!is.null(settings$las)) {
    if (length(settings$las) != 1L || !settings$las %in% 0:3) {
      stop("`las` must be 1 of 0, 1, 2, 3.", call. = FALSE)
    }
  }
  invisible(settings)
}

#' Evaluate code with temporary graphical parameters
#'
#' @param settings A named list of `par()` settings, checked with
#'   [check_par_settings()].
#' @param code Code to evaluate while the settings are active.
#' @return The value of `code`.
#' @export
with_par <- function(settings, code) {
  check_par_settings(settings)
  old <- do.call(par, settings)
  on.exit(par(old), add = TRUE)
  force(code)
}

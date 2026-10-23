#' Default parameters for a layer
#'
#' All defaults are derived from `size`, `alpha` and `width`, so a house
#' style can change them in one place.
#'
#' @param geom Geom name.
#' @param size Base mark size.
#' @param alpha Base opacity.
#' @param width Base bar and box width.
#' @return A named list of fixed aesthetics and parameters for `geom`.
#' @keywords internal
layer_defaults <- function(geom = c("point", "line", "bar", "boxplot", "smooth", "text"),
                           size = 2, alpha = 0.8, width = 0.8) {
  geom <- match.arg(geom)
  out <- switch(geom,
    point = list(
      size = size,
      alpha = alpha,
      shape = 19L
    ),
    line = list(
      linewidth = size / 4,
      alpha = alpha,
      linetype = "solid"
    ),
    bar = list(
      width = width,
      alpha = alpha,
      colour = NA
    ),
    boxplot = list(
      width = width * 0.75,
      alpha = alpha,
      outlier.size = size / 2
    ),
    smooth = list(
      linewidth = size / 4,
      alpha = alpha / 2,
      se = TRUE
    ),
    text = list(
      size = size * 1.5,
      alpha = alpha,
      vjust = -0.5
    )
  )
  return(out)
}

#' Merge user parameters over layer defaults
#'
#' @param defaults Named list from [layer_defaults()].
#' @param params Named list of user parameters; `NULL` entries remove a
#'   default.
#' @return A named list.
#' @keywords internal
merge_params <- function(defaults, params) {
  if (length(params) > 0L && (is.null(names(params)) || any(!nzchar(names(params))))) {
    stop("All layer parameters must be named.", call. = FALSE)
  }
  out <- defaults
  for (nm in names(params)) {
    out[nm] <- list(params[[nm]])
  }
  out <- out[!vapply(out, is.null, logical(1))]
  return(out)
}

#' Point layer
#'
#' Checks `mapping` and `data` before building the layer, so that
#' mistakes surface here rather than at print time.
#'
#' @param mapping Aesthetic mapping created with [ggplot2::aes()].
#' @param data Layer data, or `NULL` to inherit the plot data.
#' @param ... Fixed aesthetics and parameters, merged over
#'   [layer_defaults()].
#' @param na.rm Whether to drop missing values silently.
#' @param show.legend Whether to include the layer in legends.
#' @return A ggplot2 layer.
#' @family layer builders
#' @export
lw_points <- function(mapping = NULL, data = NULL, ..., na.rm = FALSE,
                      show.legend = NA) {
  if (!is.null(mapping) && !inherits(mapping, "uneval")) {
    stop("`mapping` must be created with `aes()`.", call. = FALSE)
  }
  if (!is.null(data) && !is.data.frame(data)) {
    stop("`data` must be a data frame or NULL.", call. = FALSE)
  }

  params <- layer_defaults("point") |>
    merge_params(list(...)) |>
    merge_params(list(na.rm = na.rm))
  layer <- ggplot2::layer(
    geom = "point",
    stat = "identity",
    position = "identity",
    mapping = mapping,
    data = data,
    params = params,
    show.legend = show.legend,
    inherit.aes = TRUE
  )
  return(layer)
}

#' Line layer
#'
#' @inheritParams lw_points
#' @return A ggplot2 layer.
#' @family layer builders
#' @export
lw_lines <- function(mapping = NULL, data = NULL, ..., na.rm = FALSE,
                     show.legend = NA) {
  if (!is.null(mapping) && !inherits(mapping, "uneval")) {
    stop("`mapping` must be created with `aes()`.", call. = FALSE)
  }
  if (!is.null(data) && !is.data.frame(data)) {
    stop("`data` must be a data frame or NULL.", call. = FALSE)
  }

  params <- layer_defaults("line") |>
    merge_params(list(...)) |>
    merge_params(list(na.rm = na.rm))
  layer <- ggplot2::layer(
    geom = "line",
    stat = "identity",
    position = "identity",
    mapping = mapping,
    data = data,
    params = params,
    show.legend = show.legend,
    inherit.aes = TRUE
  )
  return(layer)
}

#' Bar layer
#'
#' @inheritParams lw_points
#' @param position Bar position, `"stack"` or `"dodge"`.
#' @return A ggplot2 layer.
#' @family layer builders
#' @export
lw_bars <- function(mapping = NULL, data = NULL, position = c("stack", "dodge"), ...,
                    na.rm = FALSE, show.legend = NA) {
  position <- match.arg(position)
  if (!is.null(mapping) && !inherits(mapping, "uneval")) {
    stop("`mapping` must be created with `aes()`.", call. = FALSE)
  }
  if (!is.null(data) && !is.data.frame(data)) {
    stop("`data` must be a data frame or NULL.", call. = FALSE)
  }

  params <- layer_defaults("bar") |>
    merge_params(list(...)) |>
    merge_params(list(na.rm = na.rm))
  layer <- ggplot2::layer(
    geom = "bar",
    stat = "count",
    position = position,
    mapping = mapping,
    data = data,
    params = params,
    show.legend = show.legend,
    inherit.aes = TRUE
  )
  return(layer)
}

#' Box plot layer
#'
#' @inheritParams lw_points
#' @return A ggplot2 layer.
#' @family layer builders
#' @export
lw_boxes <- function(mapping = NULL, data = NULL, ..., na.rm = FALSE,
                     show.legend = NA) {
  if (!is.null(mapping) && !inherits(mapping, "uneval")) {
    stop("`mapping` must be created with `aes()`.", call. = FALSE)
  }
  if (!is.null(data) && !is.data.frame(data)) {
    stop("`data` must be a data frame or NULL.", call. = FALSE)
  }

  params <- layer_defaults("boxplot") |>
    merge_params(list(...)) |>
    merge_params(list(na.rm = na.rm))
  layer <- ggplot2::layer(
    geom = "boxplot",
    stat = "boxplot",
    position = "dodge2",
    mapping = mapping,
    data = data,
    params = params,
    show.legend = show.legend,
    inherit.aes = TRUE
  )
  return(layer)
}

#' Smoother layer
#'
#' @inheritParams lw_points
#' @param method Smoothing method passed to the smooth stat.
#' @return A ggplot2 layer.
#' @family layer builders
#' @export
lw_smooth <- function(mapping = NULL, data = NULL, method = "loess", ..., na.rm = FALSE,
                      show.legend = NA) {
  if (!is.null(mapping) && !inherits(mapping, "uneval")) {
    stop("`mapping` must be created with `aes()`.", call. = FALSE)
  }
  if (!is.null(data) && !is.data.frame(data)) {
    stop("`data` must be a data frame or NULL.", call. = FALSE)
  }

  params <- layer_defaults("smooth") |>
    merge_params(list(method = method, ...)) |>
    merge_params(list(na.rm = na.rm))
  layer <- ggplot2::layer(
    geom = "smooth",
    stat = "smooth",
    position = "identity",
    mapping = mapping,
    data = data,
    params = params,
    show.legend = show.legend,
    inherit.aes = TRUE
  )
  return(layer)
}

#' Build layers from a list of specs
#'
#' @description
#' Each spec is a list with a `geom` element (`"point"`, `"line"`, `"bar"`,
#' `"boxplot"` or `"smooth"`), an optional `mapping` and any further fixed
#' parameters.
#'
#' @param specs A list of layer specs.
#' @return A list of ggplot2 layers, one per spec.
#' @export
build_layers <- function(specs) {
  layers <- vector("list", length(specs))
  for (i in seq_along(specs)) {
    spec <- specs[[i]]
    args <- spec[setdiff(names(spec), "geom")]
    builder <- switch(spec$geom,
      point = lw_points,
      line = lw_lines,
      bar = lw_bars,
      boxplot = lw_boxes,
      smooth = lw_smooth,
      stop(sprintf("No layer builder for geom `%s`.", spec$geom), call. = FALSE)
    )
    layers[[i]] <- do.call(builder, args)
  }
  return(layers)
}

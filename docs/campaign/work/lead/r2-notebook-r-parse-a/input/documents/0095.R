export_svg <- function(gv) {
  # Check to make sure that V8 is available
  if (!requireNamespace("V8")) {
    stop("V8 is required to export.", call. = FALSE)
  }

  # Ensure that the minimum version of V8 is 1.0
  stopifnot(packageVersion("V8") >= "0.10")

  # Check to make sure gv is grViz
  if (!inherits(gv, "grViz")) {
    "gv must be a grViz htmlwidget."
  }

  # Create a new V8 context
  ct <- new_context("window")

  # Source the `vis.js` JS library
  invisible(ct$source(system.file(
    "htmlwidgets/lib/viz/viz.js",
    package = "DiagrammeR"
  )))

  # Create the SVG file
  svg <-
    ct$call("Viz", gv$x$diagram, "svg", gv$x$config$engine, gv$x$config$options)

  return(svg)
}
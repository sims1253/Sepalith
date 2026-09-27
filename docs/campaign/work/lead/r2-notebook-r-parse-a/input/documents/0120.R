autoplot.trace <- function(object, y, log = TRUE, ...) {
  p <- ggplot(object, aes(x = .data$.time, {{ y }})) +
    geom_point() +
    xlab("time (s)")
  if (log) {
    p <- p + scale_y_log10()
  }
  p
}
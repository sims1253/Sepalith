get_color <- function(
  x,
  colors = c("green", "black", "red"),
  fc = c(-5, 0, 5),
  gradient = function(x) x,
  ...
) {
  col_MIN <- as.vector(col2rgb(colors[1]))
  col_MEDIAN <- as.vector(col2rgb(colors[2]))
  col_MAX <- as.vector(col2rgb(colors[3]))

  fc <- sign(fc) * gradient(abs(fc))

  color <- character(length(x))
  for (i in seq_along(x)) {
    if (!is.numeric(x[i])) {
      color[i] <- rgb(128, 12, 128, maxColorValue = 255)
      next
    }
    value <- sign(x[i]) * (gradient(abs(x[i])))
    if (x[i] <= fc[2]) {
      col_num <- (value - fc[2]) *
        (col_MEDIAN - col_MIN) /
        (fc[2] - fc[1]) +
        col_MEDIAN
    } else {
      col_num <- (value - fc[2]) *
        (col_MEDIAN - col_MAX) /
        (fc[2] - fc[3]) +
        col_MEDIAN
    }

    col_num <- as.integer(col_num)
    col_num[col_num > 255] <- 255
    col_num[col_num < 0] <- 0

    color[i] <- rgb(
      col_num[1],
      col_num[2],
      col_num[3],
      maxColorValue = 255,
      ...
    )
  }

  return(color)
}
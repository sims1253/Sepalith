set.mfrow <- function(nvars = 1, overimpute = FALSE) {
  if (overimpute) {
    ## If we are overimputing as well, we need
    ## two plots per variable
    mfrow <- switch(
      min(nvars, 13),
      c(2, 1), ## 2  plot : 1x2
      c(2, 2), ## 4  plots: 2x2
      c(3, 2), ## 6  plots: 3x2
      c(4, 2), ## 8  plots: 4x2
      c(3, 2), ## 10 plots: 3x2
      c(3, 2), ## 12 plots: 3x2
      c(4, 2), ## 14 plots: 4x2
      c(4, 2), ## 16 plots: 4x2
      c(4, 2), ## 18 plots: 4x2
      c(3, 2), ## 20 plots: 3x2
      c(3, 2), ## 22 plots: 3x2
      c(3, 2), ## 24 plots: 3x2
      c(4, 2)
    ) ## 26 plots: 4x2
  } else {
    mfrow <- switch(
      min(nvars, 13),
      c(1, 1), ## 1  plot : 1x1
      c(2, 1), ## 2  plots: 2x1
      c(2, 2), ## 3  plots: 2x2
      c(2, 2), ## 4  plots: 2x2
      c(3, 2), ## 5  plots: 3x2
      c(3, 2), ## 6  plots: 3x2
      c(3, 3), ## 7  plots: 3x3
      c(3, 3), ## 8  plots: 3x3
      c(3, 3), ## 9  plots: 3x3
      c(3, 2), ## 10 plots: 3x2
      c(3, 2), ## 11 plots: 3x2
      c(3, 2), ## 12 plots: 3x2
      c(3, 3)
    ) ## 13 plots: 3x3
  }

  return(mfrow)
}
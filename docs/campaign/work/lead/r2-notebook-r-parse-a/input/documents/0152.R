reschedule <- function(matrix, mRNA, lncRNA, sncRNA, rangeMinMax) {
  for (x in 1:mRNA) {
    for (y in seq_along(matrix[1, ])) {
      matrix[x, y] <- ((matrix[x, y] - rangeMinMax[1]) /
        (rangeMinMax[2] - rangeMinMax[1]))
    }
  }
  if (lncRNA != 0) {
    for (x in (mRNA + 1):(mRNA + lncRNA)) {
      for (y in seq_along(matrix[1, ])) {
        matrix[x, y] <- ((matrix[x, y] - rangeMinMax[3]) /
          (rangeMinMax[4] - rangeMinMax[3]))
      }
    }
  }
  if (sncRNA != 0) {
    for (x in (mRNA + lncRNA + 1):(mRNA + lncRNA + sncRNA)) {
      for (y in seq_along(matrix[1, ])) {
        matrix[x, y] <- ((matrix[x, y] - rangeMinMax[5]) /
          (rangeMinMax[6] - rangeMinMax[5]))
      }
    }
  }

  return(matrix)
}
cosine <- function(CF, type = c("user", "item"), i, j) {
  # Validation ----
  if (CF$datatype != "rating") {
    stop("*** only defined for ratings data. ***")
  }

  if (!type %in% c("user", "item")) {
    stop("*** type must be 'user' or 'item' ***")
  }

  # Single comparison ----
  if (length(j) == 1) {
    if (type == "user") {
      v <- CF$MU[i, ]
      w <- CF$MU[j, ]
    } else {
      v <- CF$MU[, i]
      w <- CF$MU[, j]
    }

    s <- ifelse(
      sum(v * v, na.rm = TRUE) == 0 | sum(w * w, na.rm = TRUE) == 0,
      0,
      sum(v * w, na.rm = TRUE) /
        (sqrt(sum(v * v, na.rm = TRUE) * sum(w * w, na.rm = TRUE)))
    )

    return(s)
  } else {
    # Multiple comparisons ----
    if (type == "user") {
      v <- CF$MU[i, ]
      w <- CF$MU[j, ]

      s <- apply(w, MARGIN = 1, function(x) {
        ifelse(
          sum(v * v, na.rm = TRUE) == 0 | sum(x * x, na.rm = TRUE) == 0,
          0,
          sum(v * x, na.rm = TRUE) /
            (sqrt(sum(v * v, na.rm = TRUE) * sum(x * x, na.rm = TRUE)))
        )
      })

      return(s)
    } else {
      v <- CF$MU[, i]
      w <- CF$MU[, j]

      s <- apply(w, MARGIN = 2, function(x) {
        ifelse(
          sum(v * v, na.rm = TRUE) == 0 | sum(x * x, na.rm = TRUE) == 0,
          0,
          sum(v * x, na.rm = TRUE) /
            (sqrt(sum(v * v, na.rm = TRUE) * sum(x * x, na.rm = TRUE)))
        )
      })
    }

    return(s)
  }
}

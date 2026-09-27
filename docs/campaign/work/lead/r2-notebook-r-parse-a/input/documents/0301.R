process_I_vector <- function(incid) {
  # here, the incident counts are being forced into a vector.
  if (inherits(incid, "incidence")) {
    incid <- rowSums(incidence::get_counts(incid))
  }
  if (!is.vector(incid)) {
    if (is.data.frame(incid)) {
      if (ncol(incid) == 1) {
        incid <- as.vector(incid[, 1])
      } else if ("I" %in% names(incid)) {
        incid <- as.vector(incid$I)
      } else if (!all(c("local", "imported") %in% names(incid))) {
        stop(
          "incid must be a vector or a dataframe with at least a column named
             'I' or two columns named 'local' and 'imported'."
        )
      }
    } else {
      stop(
        "incid must be a vector or a dataframe with at least a column named 
           'I' or two columns named 'local' and 'imported'."
      )
    }
  }
  incid[which(is.na(incid))] <- 0
  date_col <- names(incid) == "dates"
  if (any(date_col)) {
    if (any(incid[, !date_col] < 0)) {
      stop("incid must contain only non negative integer values.")
    }
  } else {
    if (any(incid < 0)) {
      stop("incid must contain only non negative integer values.")
    }
  }

  return(incid)
}
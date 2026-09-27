process_si_data <- function(si_data) {
  # NULL entries
  if (is.null(si_data)) {
    stop("Method si_from_data requires non NULL argument si_data")
  }

  # wrong number of columns
  si_data <- as.data.frame(si_data)
  num_cols <- dim(si_data)[2]
  if (num_cols < 4 || num_cols > 5) {
    stop("si_data should have 4 or 5 columns")
  }

  # entries with incorrect column names
  if (!all(c("EL", "ER", "SL", "SR") %in% names(si_data))) {
    names <- c("EL", "ER", "SL", "SR", "type")
    names(si_data) <- names[seq_len(num_cols)]
    warning(
      "column names for si_data were not as expected; they were 
            automatically interpreted as 'EL', 'ER', 'SL', 'SR', and 'type' 
            (the last one only if si_data had five columns). "
    )
  }

  # non integer entries in date columns
  if (!all(vlapply(seq_len(4), function(e) class(si_data[, e]) == "integer"))) {
    stop("si_data has entries for which EL, ER, SL or SR are non integers.")
  }

  # entries with wrong order in lower and upper bounds of dates
  if (any(si_data$ER - si_data$EL < 0)) {
    stop("si_data has entries for which ER<EL.")
  }
  if (any(si_data$SR - si_data$SL < 0)) {
    stop("si_data has entries for which SR<SL.")
  }

  # entries with negative serial interval
  if (any(si_data$SR - si_data$EL <= 0)) {
    stop(
      "You cannot fit any of the supported distributions to this SI dataset, 
         because for some data points the maximum serial interval is <=0."
    )
  }

  ## check that the types [0: double censored, 1; single censored,
  ## 2: exact observation] are correctly specified, and if not present
  ## put them in.
  tmp_type <- 2 -
    rowSums(cbind(si_data$ER - si_data$EL != 0, si_data$SR - si_data$SL != 0))
  if (!("type" %in% names(si_data))) {
    warning(
      "si_data contains no 'type' column. This is inferred automatically 
            from the other columns."
    )
    si_data$type <- tmp_type
  } else if (anyNA(si_data$type) | !all(si_data$type == tmp_type)) {
    warning(
      "si_data contains unexpected entries in the 'type' column. This is 
            inferred automatically from the other columns."
    )
    si_data$type <- tmp_type
  }

  return(si_data)
}
PeriodMean <- function(data, dates = NULL, start = NULL, end = NULL,
                       time_dim = 'time', na.rm = FALSE, ncores = NULL) {
  # Initial checks
  ## data
  if (is.null(data)) {
    stop("Parameter 'data' cannot be NULL.")
  }
  if (!is.numeric(data)) {
    stop("Parameter 'data' must be numeric.")
  }
  ## time_dim
  if (!is.character(time_dim) | length(time_dim) != 1) {
    stop("Parameter 'time_dim' must be a character string.")
  }
  if (!is.array(data)) {
    dim(data) <- length(data)
    names(dim(data)) <- time_dim
  }
  if (!time_dim %in% names(dim(data))) {
    stop("Parameter 'time_dim' is not found in 'data' dimension.")
  }

  if (!is.null(start) && !is.null(end)) {
    if (is.null(dates)) {
      warning("Parameter 'dates' is NULL and the average of the ",
              "full data provided in 'data' is computed.")
    } else {
      if (!any(c(is.list(start), is.list(end)))) {
      stop("Parameter 'start' and 'end' must be lists indicating the ",
            "day and the month of the period start and end.")
      }
      if (!is.null(dim(dates))) {
        data <- SelectPeriodOnData(data = data, dates = dates, start = start, 
                                   end = end, time_dim = time_dim, 
                                   ncores = ncores)
      } else {
        warning("Parameter 'dates' must have named dimensions if 'start' and ",
                "'end' are not NULL. All data will be used.")
      }
    }
  }
  total <- Apply(list(data), target_dims = time_dim, fun = mean,
                 na.rm = na.rm, ncores = ncores)$output1
  
  dim(total) <- c(dim(total), setNames(1, time_dim))
  return(total)
}
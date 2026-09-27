CST_PeriodMean <- function(data, start = NULL, end = NULL,
                           time_dim = 'time', na.rm = FALSE,
                           ncores = NULL) {
  # Check 's2dv_cube'
  if (!inherits(data, 's2dv_cube')) {
    stop("Parameter 'data' must be of the class 's2dv_cube'.")
  }

  # Dates subset
  if (!is.null(start) && !is.null(end)) {
    if (is.null(dim(data$attrs$Dates))) {
      warning("Dimensions in 'data' element 'attrs$Dates' are missed and ",
              "all data would be used.")
      start <- NULL
      end <- NULL
    }
  }

  Dates <- data$attrs$Dates
  total <- PeriodMean(data = data$data, dates = Dates, start = start, end = end,
                      time_dim = time_dim, na.rm = na.rm, ncores = ncores)
  
  data$data <- total
  data$dims <- dim(total)
  data$coords[[time_dim]] <- 1 : length(data$dims[[time_dim]])

  if (!is.null(Dates)) {
    if (!is.null(start) && !is.null(end)) {
      Dates <- SelectPeriodOnDates(dates = Dates, start = start, end = end,
                                   time_dim = time_dim, ncores = ncores)
    }
    if (is.null(dim(Dates))) {
      warning("Element 'Dates' has NULL dimensions. They will not be ", 
              "subset and 'time_bounds' will be missed.")
      data$attrs$Dates <- Dates
    } else {
      # Create time_bounds
      time_bounds <- NULL
      time_bounds$start <- ClimProjDiags::Subset(x = Dates, along = time_dim, 
                                                 indices = 1, drop = FALSE)
      time_bounds$end <- ClimProjDiags::Subset(x = Dates, along = time_dim, 
                                               indices = dim(Dates)[time_dim], 
                                               drop = FALSE)

      # Add Dates in attrs
      data$attrs$Dates <- time_bounds$start
      data$attrs$time_bounds <- time_bounds
    }
  }
  return(data)
}
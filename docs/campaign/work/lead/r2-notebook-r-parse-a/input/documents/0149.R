read_empirical_data <- function(
  file,
  time_col = NULL,
  date_format = NULL,
  ...
) {
  # Determine file type and read
  ext <- tolower(tools::file_ext(file))

  data <- switch(
    ext,
    "csv" = utils::read.csv(file, stringsAsFactors = FALSE, ...),
    "rds" = readRDS(file),
    "rdata" = {
      env <- new.env()
      load(file, envir = env)
      obj_names <- ls(env)
      if (length(obj_names) == 1) {
        get(obj_names[1], envir = env)
      } else {
        stop("RData file contains multiple objects. Please use RDS format.")
      }
    },
    stop("Unsupported file format: ", ext)
  )

  # Auto-detect time column if not specified
  if (is.null(time_col)) {
    time_candidates <- c(
      "time",
      "t",
      "date",
      "year",
      "period",
      "Time",
      "Date",
      "Year"
    )
    time_col <- intersect(time_candidates, names(data))[1]
    if (!is.na(time_col)) {
      message("Auto-detected time column: ", time_col)
    }
  }

  # Convert date column to numeric if needed
  if (!is.null(time_col) && time_col %in% names(data)) {
    if (inherits(data[[time_col]], "Date")) {
      data[[time_col]] <- as.numeric(data[[time_col]])
    } else if (is.character(data[[time_col]])) {
      if (!is.null(date_format)) {
        data[[time_col]] <- as.numeric(as.Date(
          data[[time_col]],
          format = date_format
        ))
      } else {
        # Try common formats
        for (fmt in c("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y", "%Y")) {
          parsed <- tryCatch(
            as.Date(data[[time_col]], format = fmt),
            error = function(e) NULL
          )
          if (!is.null(parsed) && !all(is.na(parsed))) {
            data[[time_col]] <- as.numeric(parsed)
            break
          }
        }
      }
    }
  }

  # Add class for method dispatch
  class(data) <- c("empirical_data", class(data))
  attr(data, "time_col") <- time_col

  data
}
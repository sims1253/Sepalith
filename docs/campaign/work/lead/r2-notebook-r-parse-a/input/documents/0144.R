type_check <- function(data, expected) {

  #---------------------------------------------
  # Validate data
  #---------------------------------------------
  validate_data(data)

  #---------------------------------------------
  # Validate expected
  #---------------------------------------------
  if (missing(expected) ||
      !is.character(expected) ||
      length(expected) == 0L) {

    stop(
      "'expected' must be a non-empty named character vector.",
      call. = FALSE
    )
  }

  expected_names <- names(expected)

  if (is.null(expected_names) ||
      any(is.na(expected_names)) ||
      any(!nzchar(expected_names))) {

    stop(
      "'expected' must have variable names.",
      call. = FALSE
    )
  }

  if (anyDuplicated(expected_names)) {

    stop(
      "Variable names in 'expected' must be unique.",
      call. = FALSE
    )
  }

  #---------------------------------------------
  # Supported types
  #---------------------------------------------
  supported_types <- c(
    "numeric",
    "integer",
    "character",
    "factor",
    "logical",
    "Date",
    "POSIXct"
  )

  if (any(!expected %in% supported_types)) {

    invalid <- unique(
      expected[
        !expected %in% supported_types
      ]
    )

    stop(
      sprintf(
        "Unsupported expected type(s): %s.",
        paste(
          invalid,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }

  #---------------------------------------------
  # Check requested variables exist
  #---------------------------------------------
  missing_variables <- setdiff(
    expected_names,
    names(data)
  )

  if (length(missing_variables) > 0L) {

    stop(
      sprintf(
        "Variable(s) not found in 'data': %s.",
        paste(
          missing_variables,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }

  #---------------------------------------------
  # Determine actual type
  #---------------------------------------------
  actual_type <- function(x) {

    if (inherits(x, "Date")) {
      return("Date")
    }

    if (inherits(x, "POSIXct")) {
      return("POSIXct")
    }

    if (is.factor(x)) {
      return("factor")
    }

    if (is.logical(x)) {
      return("logical")
    }

    if (is.integer(x)) {
      return("integer")
    }

    if (is.numeric(x)) {
      return("numeric")
    }

    if (is.character(x)) {
      return("character")
    }

    class(x)[1L]
  }

  #---------------------------------------------
  # Type matching
  #---------------------------------------------
  matches_type <- function(x, expected_type) {

    switch(
      expected_type,

      numeric = is.numeric(x),

      integer = is.integer(x),

      character = is.character(x),

      factor = is.factor(x),

      logical = is.logical(x),

      Date = inherits(
        x,
        "Date"
      ),

      POSIXct = inherits(
        x,
        "POSIXct"
      )
    )
  }

  #---------------------------------------------
  # Extract requested variables
  #---------------------------------------------
  actual <- vapply(
    expected_names,
    function(variable) {

      actual_type(
        data[[variable]]
      )

    },
    character(1)
  )

  match <- vapply(
    seq_along(expected),
    function(i) {

      matches_type(
        data[[expected_names[i]]],
        expected[i]
      )

    },
    logical(1)
  )

  #---------------------------------------------
  # Construct result
  #---------------------------------------------
  result <- data.frame(
    Variable = unname(expected_names),
    Expected = unname(expected),
    Actual = unname(actual),
    Match = unname(match),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  #---------------------------------------------
  # Class
  #---------------------------------------------
  class(result) <- c(
    "TypeCheck",
    "data.frame"
  )

  result
}
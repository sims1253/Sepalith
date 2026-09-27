build_design_matrix <- function(
  x_t,
  dt = 1,
  sg_poly_order = 4,
  library_degree = 5,
  library_type = c("poly", "four", "poly_four")
) {
  monomial_degree <- library_degree
  dt <- dt
  # Filter x_t
  num_columns <- ncol(x_t)
  x_filtered <- list()
  xdot_filtered <- list()
  # Filter x_t
  for (i in 1:num_columns) {
    sg_combinations <- sg_optimal_combination(
      x_t[, i],
      dt,
      polyorder = sg_poly_order
    )[[2]]
    x_filtered[[i]] <- sgolayfilt(
      x_t[, i],
      p = sg_combinations[1, 1],
      n = sg_combinations[1, 2],
      m = 0,
      ts = dt
    )
    xdot_filtered[[i]] <- sgolayfilt(
      x_t[, i],
      p = sg_combinations[1, 1],
      n = sg_combinations[1, 2],
      m = 1,
      ts = dt
    )
  }
  # Combine filtered data and derivatives
  x_t <- do.call(cbind, x_filtered)
  sg_dx <- do.call(cbind, xdot_filtered)
  # Get the number of columns in the matrix
  num_columns_sg_dx <- ncol(sg_dx)
  # Create column names based on the pattern
  colnames(sg_dx) <- paste0("xdot", 1:num_columns_sg_dx)
  ### Sort state variables for expansion
  ### x_t needs to be in reverse order because of how poly function expands
  ### We do this here so that we can use it for the for loop to determine
  ### optimal SG parameters
  out_sorted <- x_t %>%
    data.frame() %>%
    rev()
  if (library_type == "poly" | library_type == "poly_four") {
    # Polynomial Expansion
    expanded_theta <- polym(
      as.matrix(out_sorted),
      degree = monomial_degree,
      raw = TRUE
    )
    # Order by degree using as.numeric_version numeric_version allows to
    # convert names of variables and expand without limit
    ordered_results <- order(
      attr(expanded_theta, "degree"),
      as.numeric_version(colnames(expanded_theta))
    )
    # Sort Theta Matrix
    sorted_theta <- expanded_theta[, ordered_results]
    sorted_theta <- data.frame(sorted_theta)
    # Change Variable Names
    s <- strsplit(substring(colnames(sorted_theta), 2), "\\.")
    colnames(sorted_theta) <- sapply(s, function(powers) {
      terms <- mapply(
        function(power, index) {
          if (power == "0") {
            return(NULL)
          } else if (power == "1") {
            return(paste0("x", index))
          } else {
            return(paste0("x", index, "^", power))
          }
        },
        powers,
        rev(seq_along(powers)),
        SIMPLIFY = FALSE
      )

      # Filter out any NULL values from the terms list
      terms <- Filter(Negate(is.null), terms)

      # Sort terms alphabetically
      sorted_terms <- sort(unlist(terms))

      # Collapse the sorted terms into one string
      paste(sorted_terms, collapse = "")
    })
    # That lost the attributes, so put them back
    attr(sorted_theta, "degree") <-
      attr(expanded_theta, "degree")[ordered_results]
    monomial_orders <-
      attr(expanded_theta, 'degree')[ordered_results]
  }
  if (library_type == "four" | library_type == "poly_four") {
    if (ncol(x_t) == 1) {
      trig_functions <- cbind(sin(x_t[, 1]), cos(x_t[, 1]))
      attr(trig_functions, "degree") <- c(1, 1)
    } else if (ncol(x_t) == 2) {
      trig_functions <- cbind(
        sin(x_t[, 1]),
        cos(x_t[, 1]),
        sin(x_t[, 2]),
        cos(x_t[, 2])
      )
      attr(trig_functions, "degree") <- c(1, 1, 1, 1)
    } else {
      trig_functions <- cbind(
        sin(x_t[, 1]),
        cos(x_t[, 1]),
        sin(x_t[, 2]),
        cos(x_t[, 2]),
        sin(x_t[, 3]),
        cos(x_t[, 3])
      )
      attr(trig_functions, "degree") <- c(1, 1, 1, 1, 1, 1)
    }
    num_columns <- ncol(trig_functions)
    column_names <- character(num_columns)

    for (i in seq(1, num_columns, by = 2)) {
      # sin for odd columns
      column_names[i] <- paste("sin_x", ceiling(i / 2), sep = "")

      # If there's an even column left
      if (i + 1 <= num_columns) {
        column_names[i + 1] <- paste("cos_x", ceiling(i / 2), sep = "")
      }
    }
    colnames(trig_functions) <- column_names
    if (library_type == "four") {
      sorted_theta <- trig_functions
      attr(sorted_theta, "degree") <-
        attr(sorted_theta, "degree")[c(attr(trig_functions, "degree"))]
      # That lost the attributes again, so put them back
      monomial_orders <-
        attr(trig_functions, "degree")
    } else {
      sorted_theta <- cbind(trig_functions, sorted_theta)
      attr(sorted_theta, "degree") <-
        attr(expanded_theta, "degree")[c(
          attr(trig_functions, "degree"),
          ordered_results
        )]
      # That lost the attributes again, so put them back
      monomial_orders <-
        attr(expanded_theta, "degree")[c(
          attr(trig_functions, "degree"),
          ordered_results
        )]
    }
  }
  return(list(
    sorted_theta = cbind(sorted_theta),
    monomial_orders = monomial_orders,
    xdot_filtered = sg_dx
  ))
}
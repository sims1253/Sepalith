#' Build a cell key from several stratification variables
#'
#' Pastes the values of the stratification variables together so that each
#' combination identifies one post-stratification cell.
#'
#' @param data A data frame with one row per unit.
#' @param strata Character vector of column names defining the cells.
#' @param sep Separator placed between the variable values.
#' @return A character vector with one cell key per row of `data`.
#' @noRd
cell_key <- function(data, strata, sep = ":") {
  missing_vars <- setdiff(strata, names(data))
  if (length(missing_vars) > 0) {
    stop("Unknown strata: ", paste(missing_vars, collapse = ", "), call. = FALSE)
  }

  parts <- lapply(strata, function(s) as.character(data[[s]]))
  return(do.call(paste, c(parts, sep = sep)))
}

#' Count sample and population units per cell
#'
#' `sample_cell_totals()` sums the design weights of the sample within each
#' post-stratification cell; `population_cell_totals()` does the same for the
#' counts of a population table (for example census counts by age and region).
#'
#' @param data A data frame with one row per unit.
#' @param strata Character vector of column names defining the cells.
#' @param weights Numeric vector of weights, one per row of `data`.
#' @return A named numeric vector of (weighted) counts per cell.
#' @keywords internal
sample_cell_totals <- function(data, strata, weights) {
  keys <- cell_key(data, strata)
  totals <- tapply(weights, keys, sum, na.rm = TRUE)
  totals[is.na(totals)] <- 0
  return(totals)
}

#' @rdname sample_cell_totals
#' @param population A data frame of population counts, one row per cell or
#'   finer.
#' @param count_col Name of the column in `population` holding the counts.
#' @keywords internal
population_cell_totals <- function(population, strata, count_col = "n") {
  if (!count_col %in% names(population)) {
    stop("Column `", count_col, "` not found in `population`.", call. = FALSE)
  }

  keys <- cell_key(population, strata)
  totals <- tapply(population[[count_col]], keys, sum, na.rm = TRUE)
  totals[is.na(totals)] <- 0
  return(totals)
}

#' Collapse sparse post-stratification cells
#'
#' Cells whose sample size falls below `min_n` are merged into the neighbouring
#' cell with the largest sample size that shares all but the last
#' stratification variable.
#'
#' @param data A data frame with one row per unit.
#' @param strata Character vector of column names defining the cells.
#' @param min_n Minimum number of respondents required per cell.
#' @return A character vector of (possibly collapsed) cell keys, one per row.
#' @export
collapse_sparse_cells <- function(data, strata, min_n = 20L) {
  keys <- cell_key(data, strata)
  if (length(strata) < 2) {
    return(keys)
  }

  # Siblings are the cells that agree on every stratum except the last one.
  parent <- cell_key(data, strata[-length(strata)])
  sizes <- table(keys)
  sparse <- names(sizes)[sizes < min_n]

  for (cell in sparse) {
    rows <- which(keys == cell)
    # An earlier merge may already have emptied or filled this cell.
    if (length(rows) == 0 || sizes[cell] >= min_n) {
      next
    }
    siblings <- unique(keys[parent == parent[rows[1]] & keys != cell])
    if (length(siblings) == 0) {
      warning("Cell `", cell, "` is sparse and has no sibling to merge into.",
        call. = FALSE
      )
      next
    }

    sibling_sizes <- sizes[siblings]
    target <- names(sibling_sizes)[which.max(sibling_sizes)]
    keys[rows] <- target
    sizes[target] <- sizes[target] + length(rows)
    sizes[cell] <- 0
  }

  return(keys)
}

#' Compute post-stratification adjustment factors
#'
#' Divides the population count of each cell by the weighted sample count.
#' Cells present in the population but not the sample are reported.
#'
#' @param sample_totals Named numeric vector from [sample_cell_totals()].
#' @param population_totals Named numeric vector from
#'   [population_cell_totals()].
#' @return A named numeric vector of adjustment factors, one per sample cell.
#' @keywords internal
poststrat_factors <- function(sample_totals, population_totals) {
  empty <- setdiff(names(population_totals), names(sample_totals))
  if (length(empty) > 0) {
    warning(length(empty), " population cell(s) have no sample: ",
      paste(utils::head(empty, 5), collapse = ", "),
      call. = FALSE
    )
  }

  # A sample cell with no population count has no defensible factor at all.
  unknown <- setdiff(names(sample_totals), names(population_totals))
  if (length(unknown) > 0) {
    stop("Sample cells missing from population: ",
      paste(unknown, collapse = ", "),
      call. = FALSE
    )
  }

  factors <- population_totals[names(sample_totals)] / sample_totals
  factors[!is.finite(factors)] <- 0
  return(factors)
}

#' Post-stratify survey weights by hand
#'
#' Scales design weights so that the weighted sample reproduces the population
#' count in every cell defined by `strata`.
#'
#' @param data A data frame with one row per unit.
#' @param population A data frame of population counts, one row per cell or
#'   finer.
#' @param strata Character vector of column names defining the cells.
#' @param weights Numeric vector of weights, one per row of `data`.
#' @param count_col Name of the column in `population` holding the counts.
#' @return A numeric vector of post-stratified weights with an attribute
#'   `factors` holding the per-cell adjustment.
#' @export
poststratify <- function(data, population, strata, weights = NULL, count_col = "n") {
  if (is.null(weights)) {
    weights <- rep(1, nrow(data))
  }

  sample_totals <- sample_cell_totals(data, strata, weights)
  population_totals <- population_cell_totals(population, strata, count_col)
  factors <- poststrat_factors(sample_totals, population_totals)

  keys <- cell_key(data, strata)
  adjusted <- weights * unname(factors[keys])
  attr(adjusted, "factors") <- factors
  return(adjusted)
}

roi_filter_threshold <- function(img_obj, threshold, use_processed = FALSE) {
  # 1. Validate Input
  if (!inherits(img_obj, "BioThermR")) {
    stop("Error: Input must be a 'BioThermR' object.")
  }

  if (!is.numeric(threshold) || length(threshold) != 2) {
    stop(
      "Error: 'threshold' must be a numeric vector of length 2, e.g., c(20, 35)."
    )
  }

  # Ensure the range is sorted (min, max)
  threshold <- sort(threshold)
  min_val <- threshold[1]
  max_val <- threshold[2]

  mat <- if (use_processed) img_obj$processed else img_obj$raw

  # 2. Create Mask
  # Keep values that are >= min AND <= max
  # We wrap in () to ensure order of operations
  mask <- (mat >= min_val) & (mat <= max_val)

  # Handle existing NAs (NA comparison results in NA, we want FALSE)
  mask[is.na(mask)] <- FALSE

  # 3. Apply Mask
  # Set everything NOT in the mask to NA
  mat[!mask] <- NA

  # 4. Update Object
  img_obj$processed <- mat
  img_obj$stats <- NULL

  message(paste0(
    "ROI Filter applied: Keeping range [",
    min_val,
    ", ",
    max_val,
    "]"
  ))

  return(img_obj)
}
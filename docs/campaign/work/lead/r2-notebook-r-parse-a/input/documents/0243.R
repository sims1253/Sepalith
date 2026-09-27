roi_segment_ebimage <- function(
  img_obj,
  method = "otsu",
  keep_largest = TRUE,
  morphology = TRUE
) {
  if (!inherits(img_obj, "BioThermR")) {
    stop("Error: Input must be a 'BioThermR' object.")
  }

  # Check for EBImage dependency
  if (!requireNamespace("EBImage", quietly = TRUE)) {
    stop(
      "Error: Package 'EBImage' is required. Please install it via BiocManager::install('EBImage')."
    )
  }

  mat <- img_obj$processed

  # Create a temporary normalized image
  range_val <- range(mat, na.rm = TRUE)
  mat_norm <- (mat - range_val[1]) / (range_val[2] - range_val[1])

  # Convert to EBImage object
  eb_img <- EBImage::Image(mat_norm)

  # 1. Thresholding
  if (method == "otsu") {
    # Calculate Otsu's threshold
    thr <- EBImage::otsu(eb_img)
    mask <- mat_norm > thr
  } else {
    stop("Error: Only 'otsu' method is currently supported.")
  }

  # 2. Morphological Operations (Optional)
  if (morphology) {
    kern <- EBImage::makeBrush(5, shape = "disc")
    mask <- EBImage::opening(mask, kern)
    mask <- EBImage::closing(mask, kern)
  }

  # 3. Connected Components Labeling (Find objects)
  # Converts boolean mask to integer labels (0=bg, 1=obj1, 2=obj2...)
  labels <- EBImage::bwlabel(mask)

  # 4. Keep Largest Object
  if (keep_largest) {
    # Count pixels per label
    tbl <- table(labels)
    # Remove background (label 0)
    tbl <- tbl[names(tbl) != "0"]

    if (length(tbl) > 0) {
      # Find ID of the largest object
      largest_id <- names(which.max(tbl))
      # Update mask to only include this object
      mask <- (labels == as.integer(largest_id))
      message(paste(
        "Auto-Segmentation: Kept largest object (",
        max(tbl),
        "pixels )"
      ))
    } else {
      warning("Warning: No object detected after thresholding.")
      mask <- matrix(FALSE, nrow = nrow(mat), ncol = ncol(mat))
    }
  }

  # 5. Apply Mask to Original Matrix
  # Convert EBImage mask back to logical matrix
  final_mask <- as.matrix(mask)

  # Set background to NA
  img_obj$processed[!final_mask] <- NA
  img_obj$stats <- NULL # Reset stats

  return(img_obj)
}
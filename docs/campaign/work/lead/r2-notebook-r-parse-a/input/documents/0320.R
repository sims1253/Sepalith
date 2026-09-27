get_countsMADC <- function(
  madc_file = NULL,
  madc_object = NULL,
  collapse_matches_counts = FALSE,
  verbose = TRUE
) {
  # Add check inputs
  if (is.null(madc_file) && is.null(madc_object)) {
    stop("Please provide either madc_file or madc_object.")
  }
  if (!is.null(madc_file) && !is.null(madc_object)) {
    stop("Please provide either madc_file or madc_object. Not both.")
  }
  if (!is.null(madc_file) && !file.exists(madc_file)) {
    stop("MADC file not found. Please provide a valid path.")
  }
  if (!is.null(madc_object) && !is.data.frame(madc_object)) {
    stop("madc_object must be a data frame.")
  }

  vmsg(
    paste0(
      "Extracting read counts from ",
      ifelse(
        !is.null(madc_file),
        paste0("MADC file: ", madc_file),
        "madc_object"
      )
    ),
    verbose = verbose,
    level = 0,
    type = ">>"
  )
  vmsg(
    ifelse(
      collapse_matches_counts,
      "|AltMatch and |RefMatch counts will be collapsed into their respective |Ref and |Alt alleles.",
      "|AltMatch and |RefMatch rows will be discarded (collapse_matches_counts = FALSE)."
    ),
    verbose = verbose,
    level = 1,
    type = ">>"
  )

  # This function takes the MADC file as input and generates a Ref and Alt counts dataframe as output
  if (is.null(madc_object)) {
    update_df <- get_counts(
      madc_file = madc_file,
      collapse_matches_counts = collapse_matches_counts,
      verbose = verbose
    )
  } else {
    update_df <- get_counts(
      madc_object = madc_object,
      collapse_matches_counts = collapse_matches_counts,
      verbose = verbose
    )
  }
  # Ensure plain data.frame so row.names<- does not trigger tibble deprecation warning
  update_df <- as.data.frame(update_df)

  # Filter rows where 'AlleleID' ends with 'Ref'
  ref_df <- subset(update_df, grepl("Ref$", AlleleID))

  # Filter rows where 'AlleleID' ends with 'Alt'
  alt_df <- subset(update_df, grepl("Alt$", AlleleID))

  #Ensure that each has the same SNPs and that they are in the same order
  same <- identical(alt_df$CloneID, ref_df$CloneID)

  ###Convert the ref and alt counts into matrices with the CloneID as the index
  #Set SNP names as index
  row.names(ref_df) <- ref_df$CloneID
  row.names(alt_df) <- alt_df$CloneID

  #Retain only the rows in common if they are not identical and provide warning
  if (!same) {
    # Find the common CloneIDs between the two dataframes
    all_mks <- unique(c(rownames(ref_df), rownames(alt_df)))
    common_ids <- intersect(rownames(ref_df), rownames(alt_df))
    n_singles <- length(all_mks) - length(common_ids)

    vmsg(
      paste(
        "There are",
        n_singles,
        "Ref tags without corresponding Alt tags, or vice versa"
      ),
      verbose = verbose,
      level = 2,
      type = ">>"
    )
    vmsg(
      "Only the markers with both Ref and Alt tags will be retained for the conversion",
      verbose = verbose,
      level = 1,
      type = ">>"
    )
    vmsg(
      "Consider providing a haplotype database file to resolve unpaired Ref/Alt sequences",
      verbose = verbose,
      level = 1,
      type = ">>"
    )

    warning(paste(
      "There are",
      n_singles,
      "Ref tags without corresponding Alt tags, or vice versa. Only the markers with both Ref and Alt tags will be retained for the conversion. Consider providing a haplotype database file to resolve unpaired Ref/Alt sequences."
    ))

    # Subset both dataframes to retain only the common rows
    ref_df <- ref_df[common_ids, ]
    alt_df <- alt_df[common_ids, ]
  }

  #Define columns to remove
  columns_to_remove <- c(
    "AlleleID",
    "CloneID",
    "AlleleSequence",
    "ClusterConsensusSequence",
    "CallRate",
    "OneRatioRef",
    "OneRatioSnp",
    "FreqHomRef",
    "FreqHomSnp",
    "FreqHets",
    "PICRef",
    "PICSnp",
    "AvgPIC",
    "AvgCountRef",
    "AvgCountSnp",
    "RatioAvgCountRefAvgCountSnp"
  ) #Adjust as needed for different MADC versions

  #Identify columns that are in MADC
  col_remove <- intersect(columns_to_remove, colnames(ref_df))

  #Remove the specified columns and convert to matrix
  ref_matrix <- as.matrix(select(ref_df, -all_of(col_remove)))
  alt_matrix <- as.matrix(select(alt_df, -all_of(col_remove)))

  #Convert elements to numeric
  class(ref_matrix) <- "numeric"
  class(alt_matrix) <- "numeric"

  #Make the size matrix by combining the two matrices
  size_matrix <- (ref_matrix + alt_matrix)

  #Count the number of cells with 0 count to estimate missing data
  # Count the number of cells with the value 0
  count_zeros <- sum(size_matrix == 0)

  # Print the result
  ratio_missing_data <- count_zeros / length(size_matrix)
  vmsg(
    paste0(
      "Percentage of missing data (datapoints with 0 total count): ",
      round(ratio_missing_data * 100, 2),
      "%"
    ),
    verbose = verbose,
    level = 2,
    type = ">>"
  )

  # Return the ref and alt matrices as a list
  matrices_list <- list(ref_matrix = ref_matrix, size_matrix = size_matrix)

  return(matrices_list)
}
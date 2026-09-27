get_counts <- function(
  madc_file = NULL,
  madc_object = NULL,
  collapse_matches_counts = FALSE,
  verbose = TRUE
) {
  # Add check inputs
  if (is.null(madc_file) && is.null(madc_object)) {
    stop("Please provide either madc_file or madc_object.")
  }
  if (!is.null(madc_file) && !file.exists(madc_file)) {
    stop("MADC file not found. Please provide a valid path.")
  }
  if (!is.null(madc_object) && !is.data.frame(madc_object)) {
    stop("madc_object must be a data frame.")
  }

  # Read the MADC file

  if (!is.null(madc_file)) {
    #Read only the first column for the first seven rows
    first_seven_rows <- read.csv(
      madc_file,
      header = FALSE,
      nrows = 7,
      colClasses = c(NA, "NULL")
    )

    #Check if all entries in the first column are either blank or "*"
    check_entries <- all(first_seven_rows[, 1] %in% c("", "*"))
  } else {
    check_entries <- all(
      madc_object[1:min(7L, nrow(madc_object)), 1] %in% c("", "*")
    )
  }

  #Check if the MADC file has the filler rows or is processed from updated fixed allele ID pipeline
  if (check_entries) {
    #Note: This assumes that the first 7 rows are placeholder info from DArT processing
    #Read the madc file
    vmsg(
      "Detected raw MADC format with 7-row header. Reading file while skipping the first 7 rows.",
      verbose = verbose,
      level = 1,
      type = ">>"
    )
    if (!is.null(madc_file)) {
      madc_df <- read.csv(madc_file, sep = ',', skip = 7, check.names = FALSE)
    } else {
      madc_df <- madc_object[-(1:7), ]
    }
  } else {
    #Read the madc file
    vmsg(
      "Detected fixed allele IDs MADC format",
      verbose = verbose,
      level = 1,
      type = ">>"
    )
    if (!is.null(madc_file)) {
      madc_df <- read.csv(madc_file, sep = ',', check.names = FALSE)
    } else {
      madc_df <- madc_object
    }
  }

  if (collapse_matches_counts) {
    filtered_df <- madc_df[order(madc_df$AlleleID), ] %>%
      # Keep only Ref/Alt alleles and their Match variants; drop other allele types
      filter(grepl("\\|(Ref|Alt)(Match)?(_|$)", AlleleID)) %>%
      mutate(
        Type = case_when(
          grepl("\\|Alt(Match)?(_|$)", AlleleID) ~ "Alt",
          grepl("\\|Ref(Match)?(_|$)", AlleleID) ~ "Ref"
        )
      ) %>%
      group_by(CloneID, Type) %>%
      summarise(
        AlleleID = paste0(unique(CloneID), "|", unique(Type)),
        AlleleSequence = first(AlleleSequence),
        across(where(is.numeric), ~ sum(.x, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      select(AlleleID, CloneID, AlleleSequence, everything(), -Type)
  } else {
    #Retain only the Ref and Alt haplotypes
    filtered_df <- madc_df[
      !grepl("\\|AltMatch|\\|RefMatch", madc_df$AlleleID),
    ]
  }

  #Remove extra text after Ref and Alt (_001 or _002)
  filtered_df$AlleleID <- sub("\\|Ref.*", "|Ref", filtered_df$AlleleID)
  filtered_df$AlleleID <- sub("\\|Alt.*", "|Alt", filtered_df$AlleleID)

  return(filtered_df)
}
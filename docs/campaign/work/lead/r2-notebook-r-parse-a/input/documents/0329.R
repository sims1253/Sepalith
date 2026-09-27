subseq_func <- function(
  NT_seq,
  size_NT_regions,
  FWR1partial = FALSE,
  FWR4partial = FALSE,
  ORF_begins
) {
  ########### TOCAR
  AA_info <- translate_fun(toupper(NT_seq), ORF_begins)
  AA_seq <- AA_info[[1]]
  seq_list <- list()
  ORF <- AA_info[[2]]

  if ((AA_seq == "NO_GOOD_ORF")) {
    seq_list[
      1:(length(size_NT_regions) +
        2)
    ] <- "NO_GOOD_ORF"
  } else {
    # initialize
    if (FWR1partial) {
      size_NT_regions <- size_NT_regions[2:length(size_NT_regions)]
    }
    if (FWR4partial) {
      size_NT_regions <- size_NT_regions[
        1:c(
          length(size_NT_regions) -
            1
        )
      ]
    }
    size_NT_regions[1] <- size_NT_regions[1] - ORF + 1

    size_AA_regions <- c()
    off_set <- 0
    for (i in c(1:(length(size_NT_regions)))) {
      if ((size_NT_regions[i] - off_set) %% 3 == 0) {
        size_AA_regions <- c(
          size_AA_regions,
          (size_NT_regions[i] - off_set) / 3
        )
        off_set <- 0
      } else {
        if (i != (length(size_NT_regions))) {
          onset <- 3 - (size_NT_regions[i] - off_set) %% 3

          if (nchar(size_NT_regions[i + 1]) >= onset) {
            if (onset == 2) {
              size_AA_regions <- c(
                size_AA_regions,
                floor((size_NT_regions[i] - off_set) / 3)
              )
              off_set <- -1
            } else {
              size_AA_regions <- c(
                size_AA_regions,
                ceiling((size_NT_regions[i] - off_set) / 3)
              )
              off_set <- onset
            }
          } else {
            size_AA_regions <- c(
              size_AA_regions,
              floor(size_NT_regions[i] - off_set) / 3
            )
            off_set <- 0
          }
        } else {
          size_AA_regions <- c(
            size_AA_regions,
            floor(size_NT_regions[i] - off_set) / 3
          )
          off_set <- 0
        }
      }
    }
    # size_AA_regions <- round(size_NT_regions/3)
    # size_AA_regions[length(size_AA_regions) -
    #     1] <- floor(
    #     (size_NT_regions[length(size_AA_regions) -
    #         1]/3)
    # )
    # size_AA_regions[length(size_AA_regions)] <- floor((size_NT_regions[length(size_AA_regions)]/3))
    end <- 0

    for (i in 1:(length(size_AA_regions) -
      1)) {
      start <- end + 1
      end <- start - 1 + size_AA_regions[i]
      seq_list[i] <- Biostrings::subseq(AA_seq, start, end)
    }

    seq_list[[length(size_AA_regions)]] <- Biostrings::subseq(
      AA_seq,
      sum(
        size_AA_regions[
          seq_along(size_AA_regions) -
            1
        ]
      ) +
        1,
      stringr::str_length(AA_seq)
    )

    if (FWR1partial) {
      seq_list <- append(seq_list, "", after = 0)
    }

    if (FWR4partial) {
      seq_list <- append(seq_list, "", after = length(seq_list))
    }

    if (paste(unlist(seq_list), collapse = "") != AA_seq) {
      strange <- TRUE
    } else {
      strange <- FALSE
    }
    seq_list[
      length(seq_list) +
        1
    ] <- AA_seq
    seq_list[
      length(seq_list) +
        1
    ] <- ORF

    if (strange) {
      seq_list[
        length(seq_list) +
          1
      ] <- ("MMMMMMMMMMMmmmmmmmmmmm strange")
    }
  }
  return(seq_list)
}
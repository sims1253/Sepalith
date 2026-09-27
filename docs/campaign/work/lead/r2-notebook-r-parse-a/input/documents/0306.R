BEDMatrix <- function(path, n = NULL, p = NULL, simple_names = FALSE) {
  path <- path.expand(path)
  if (!file.exists(path)) {
    # Try to add extension (common in PLINK)
    path <- paste0(path, ".bed")
    if (!file.exists(path)) {
      stop("File not found.", call. = FALSE)
    }
  }
  pathSansExt <- tools::file_path_sans_ext(path)
  filesetName <- basename(pathSansExt)
  if (is.null(n)) {
    # Check if .fam file exists
    famPath <- paste0(pathSansExt, ".fam")
    if (!file.exists(famPath)) {
      stop(
        filesetName,
        ".fam not found. Provide number of samples (n).",
        call. = FALSE
      )
    } else {
      message(
        "Extracting number of samples and rownames from ",
        filesetName,
        ".fam..."
      )
      if (requireNamespace("data.table", quietly = TRUE)) {
        if (simple_names) {
          famColumns <- c(2L)
        } else {
          famColumns <- c(1L, 2L)
        }
        fam <- data.table::fread(
          famPath,
          select = famColumns,
          colClasses = list(character = famColumns),
          data.table = FALSE,
          showProgress = FALSE
        )
        # Determine n
        n <- nrow(fam)
        # Determine rownames
        if (simple_names) {
          # Use within-family ID only
          rownames <- fam[, 1L]
        } else {
          # Concatenate family ID and within-family ID
          rownames <- paste0(fam[, 1L], "_", fam[, 2L])
        }
      } else {
        fam <- readLines(famPath) # much faster than read.table
        # Determine n
        n <- length(fam)
        # Determine rownames
        if (simple_names) {
          # Use within-family ID only
          rownames <- vapply(strsplit(fam, delims), `[`, "", 2L)
        } else {
          rownames <- vapply(
            strsplit(fam, delims),
            function(line) {
              # Concatenate family ID and within-family ID
              paste0(line[1L], "_", line[2L])
            },
            ""
          )
        }
      }
    }
  } else {
    n <- as.integer(n)
    rownames <- NULL
  }
  if (is.null(p)) {
    # Check if .bim file exists
    bimPath <- paste0(pathSansExt, ".bim")
    if (!file.exists(bimPath)) {
      stop(
        filesetName,
        ".bim not found. Provide number of variants (p).",
        .call = FALSE
      )
    } else {
      message(
        "Extracting number of variants and colnames from ",
        filesetName,
        ".bim..."
      )
      if (requireNamespace("data.table", quietly = TRUE)) {
        if (simple_names) {
          bimColumns <- c(2L)
        } else {
          bimColumns <- c(2L, 5L)
        }
        bim <- data.table::fread(
          bimPath,
          select = bimColumns,
          colClasses = list(character = bimColumns),
          data.table = FALSE,
          showProgress = FALSE
        )
        # Determine p
        p <- nrow(bim)
        # Determine colnames
        if (simple_names) {
          # Use variant name only
          colnames <- bim[, 1L]
        } else {
          # Concatenate variant name and A1 allele (like '--recode A'
          # in PLINK)
          colnames <- paste0(bim[, 1L], "_", bim[, 2L])
        }
      } else {
        bim <- readLines(bimPath) # much faster than read.table
        # Determine p
        p <- length(bim)
        # Determine colnames
        if (simple_names) {
          # Use variant name only
          colnames <- vapply(strsplit(bim, delims), `[`, "", 2L)
        } else {
          colnames <- vapply(
            strsplit(bim, delims),
            function(line) {
              # Concatenate variant name and A1 allele (like
              # '--recode A' in PLINK)
              paste0(line[2L], "_", line[5L])
            },
            ""
          )
        }
      }
    }
  } else {
    p <- as.integer(p)
    colnames <- NULL
  }
  obj <- new(
    "BEDMatrix",
    xptr = .Call(C_BEDMatrix_initialize, path, n, p),
    path = path,
    dims = c(n, p),
    dnames = list(rownames, colnames)
  )
  return(obj)
}
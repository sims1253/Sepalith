verifyped <- function(pedigree, unk = 0) {
  flag <- FALSE
  if (ncol(pedigree) != 3) {
    print("Data with more than 3 columns, please verify")
    flag <- TRUE
    return(flag)
  }

  if (length(unique(pedigree[, 1])) < nrow(pedigree)) {
    print("Data with repeated entry, please verify the following entries lines")
    print(which(duplicated(pedigree[, 1]), arr.ind = TRUE))
    flag <- TRUE
    return(flag)
  }

  #Treating all as numeric
  ind.data <- as.vector(pedigree[, 1])
  sire.data <- as.vector(pedigree[, 2])
  dire.data <- as.vector(pedigree[, 3])
  sire <- match(sire.data, c(ind.data, "0"))
  dire <- match(dire.data, c(ind.data, "0"))
  ind <- as.vector(c(seq_along(ind.data)))

  missing <- c()
  #Verify the individual w/ same name in sire/dire
  missing$conflict <- c(which(sire == ind), which(dire == ind))
  if (length(missing$conflict) > 0) {
    print(
      "The following rows have the individual name equals to the parental name. Please verify."
    )
    print(pedigree[missing$conflict, ])
    flag <- TRUE
  }

  #Verify the missing sire (Parent 1)
  missing$sire.na <- c(which(is.na(sire)))
  if (length(missing$sire) > 0) {
    print(
      "The following rows have the parental 1 name (column 2) missing in the pedigree. Please verify."
    )
    print(pedigree[missing$sire, ])
    flag <- TRUE
  }

  #Verify the missing dire (Parent 2)
  missing$dire.na <- c(which(is.na(dire)))
  if (length(missing$dire) > 0) {
    print(
      "The following rows have the parental 2 name (column 3) missing in the pedigree. Please verify."
    )
    print(pedigree[missing$dire, ])
    flag <- TRUE
  }
  return(flag)
}
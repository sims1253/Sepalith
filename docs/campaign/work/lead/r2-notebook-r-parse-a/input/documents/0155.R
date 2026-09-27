GaussSmoothArray <- function(
  x,
  voxdim = c(1, 1, 1),
  ksize = 5,
  sigma = diag(3, 3),
  mask = NULL,
  var.norm = FALSE
) {
  filtmat <- GaussSmoothKernel(voxdim, ksize, sigma)

  if (!is.array(x)) {
    return("x should be an array")
  }
  if (length(dim(x)) != 3 && length(dim(x)) != 4) {
    return("array x should be 3D or 4D")
  }
  tmp <- FALSE
  if (length(dim(x)) == 3) {
    x <- array(x, dim = c(dim(x), 1))
    tmp <- TRUE
  }
  if (is.null(mask)) {
    mask <- array(1, dim = dim(x)[1:3])
  }

  if (var.norm) {
    d <- .Fortran(
      "gaussfilter2",
      as.double(x),
      as.integer(dim(x)[1]),
      as.integer(dim(x)[2]),
      as.integer(dim(x)[3]),
      as.integer(dim(x)[4]),
      as.double(filtmat),
      as.integer(ksize),
      as.double(mask),
      double(length(x)),
      PACKAGE = "AnalyzeFMRI"
    )
    c1 <- array(d[[9]], dim = dim(x))
    if (tmp) c1 <- c1[,,, 1]
  } else {
    d <- .Fortran(
      "gaussfilter1",
      as.double(x),
      as.integer(dim(x)[1]),
      as.integer(dim(x)[2]),
      as.integer(dim(x)[3]),
      as.integer(dim(x)[4]),
      as.double(filtmat),
      as.integer(ksize),
      as.double(mask),
      as.double(x),
      PACKAGE = "AnalyzeFMRI"
    )
    c1 <- array(d[[9]], dim = dim(x))
    if (tmp) c1 <- c1[,,, 1]
  }

  return(c1)
}
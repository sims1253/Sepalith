#=======================================================================
# Print
#=======================================================================
#-----------------------------------------------------------------------
# Print plotSummaryAlphaPart
#-----------------------------------------------------------------------
#' @rdname print.plotSummaryAlphaPart
#' @method print plotSummaryAlphaPart
#' @title Print a plot generate by the function
#'   \code{plotSummaryAlphaPart}
#' @usage \method{print}{plotSummaryAlphaPart}(x, ask, ...)
#' @aliases print.plotSummaryAlphaPart
#' @description Plot output object from
#'   \code{\link[AlphaPart]{plot.summaryAlphaPart}}.
#' @seealso \code{\link[AlphaPart]{plot.summaryAlphaPart}}
#'
#' @param x plotSummaryAlphaPart, output object from
#'   \code{\link[AlphaPart]{plot.summaryAlphaPart}} function
#' @param ask Logical, ask before printing another plot?
#' @param ...  Arguments passed to other functions (not used at the
#'   moment).
#'
#' @example inst/examples/examples_print.plotSummaryAlphaPart.R
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#'
#' @export

print.plotSummaryAlphaPart <- function(x, ask = interactive(), ...) {
  k <- 1
  nT <- length(x)
  for (i in 1:nT) {
    print(x[[i]])
    if (ask) {
      if (k < nT) {
        msg <- paste(
          "Press any key to print out the next plot (",
          k,
          "/",
          nT,
          ") ...\n",
          sep = ""
        )
        mp <- readline(prompt = msg)
      }
    }
    k <- k + 1
  }
}

#-----------------------------------------------------------------------
# Print summaryAlphaPart
#-----------------------------------------------------------------------
#' @title Print method for objects of the class summaryAlphaPart.
#' @rdname print.summaryAlphaPart
#' @method print summaryAlphaPart
#' @usage \method{print}{summaryAlphaPart}(x, ...)
#' @aliases print.summaryAlphaPart
#' @description Print method for objects of the class
#'   \code{summaryAlphaPart} (result of \code{summary(AlphaPart(...))}).
#'
#' @seealso
#' \code{\link[AlphaPart]{summary.AlphaPart}}
#'
#' @param x summaryAlphaPart, output object from
#'   \code{\link[AlphaPart]{summary.AlphaPart}} function.
#' @param ...  Arguments passed to other functions (not used at the
#'   moment).
#'
#' @example inst/examples/examples_summary.AlphaPart.R
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#'
#' @export

print.summaryAlphaPart <- function(x, ...) {
  nI <- nrow(x[[1]])
  cat("\n\n Summary of partitions of breeding values \n")
  cat(
    "   - paths: ",
    x$info$nP,
    " (",
    paste(x$info$lP, collapse = ", "),
    ")\n",
    sep = ""
  )
  cat(
    "   - traits: ",
    x$info$nT,
    " (",
    paste(x$info$lT, collapse = ", "),
    ")\n",
    sep = ""
  )
  if (length(x$info$warn) > 0) {
    cat("   - warning: ", paste(x$info$warn, collapse = "\n"), "\n", sep = "")
  }

  for (trt in x$info$lT) {
    cat("\n Trait:", trt, "\n\n")
    print(x[[trt]])
  }
  cat("\n")
}

#-----------------------------------------------------------------------
# Print AlphaPart
#-----------------------------------------------------------------------
#' @title Print method for the output of AlphaPart function.
#' @rdname print.AlphaPart
#' @method print AlphaPart
#' @usage \method{print}{AlphaPart}(x, n, ...)
#' @aliases print.AlphaPart
#' @description Partitioning of breeding values if often performed on
#'   quite large datasets, which quickly fills in the whole
#'   screen. Print method therefore prints out paths, number of
#'   individuals and the first and the last few lines for each trait to
#'   quickly see what kind of data is in \code{x}.
#' @seealso \code{\link[AlphaPart]{AlphaPart}}, \code{\link{head}},
#'   \code{\link{tail}}.
#'
#' @param x AlphaPart, output object from
#'   \code{\link[AlphaPart]{AlphaPart}} function.
#' @param n Integer, number of the first and last rows in \code{x} to
#'   print out using \code{\link{head}} and \code{\link{tail}}.
#' @param ...  Arguments passed to \code{print} function.
#'
#' @example inst/examples/examples_AlphaPart.R
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom utils head
#' @importFrom utils tail
#'
#' @export

print.AlphaPart <- function(x, n = 6, ...) {
  nI <- nrow(x[[1]])
  cat("\n\n Partitions of breeding values \n")
  cat("   - individuals:", nI, "\n")
  cat(
    "   - paths: ",
    x$info$nP,
    " (",
    paste(x$info$lP, collapse = ", "),
    ")\n",
    sep = ""
  )
  cat(
    "   - traits: ",
    x$info$nT,
    " (",
    paste(x$info$lT, collapse = ", "),
    ")\n",
    sep = ""
  )
  if (length(x$info$warn) > 0) {
    cat("   - warning: ", paste(x$info$warn, collapse = "\n"), "\n", sep = "")
  }

  for (trt in x$info$lT) {
    cat("\n Trait:", trt, "\n\n")
    if (nI > n * 2) {
      print(head(x[[trt]]), ...)
      cat("...\n")
      print(tail(x[[trt]]), ...)
    } else {
      print(x[[trt]], ...)
    }
  }
  cat("\n")
}

#=======================================================================
# Summary
#=======================================================================
#' @title A function to summarize AlphaPart object.
#' @rdname summary.AlphaPart
#' @method summary AlphaPart
#' @usage \method{summary}{AlphaPart}(object, by, FUN, labelSum, subset,
#'   sums, cov,  ...)
#' @description Breedng values of individuals are often summarized,
#'   either by year of birth or some other classification. Function
#'   \code{summary.AlphaPart} provides a way to ease the computation of
#'   such summaries on partitions of breeding values.
#'
#' @seealso \code{\link[AlphaPart]{AlphaPart}} for partitioning breeding
#'   values, \code{\link[AlphaPart]{plot.summaryAlphaPart}} for plotting
#'   output of summary method
#'
#' @param object AlphaPart, output object from
#'   \code{\link[AlphaPart]{AlphaPart}} function.
#' @param by Character, the name of a column by which summary function
#'   FUN should be applied; if \code{NULL} (default) summary is given
#'   for the whole table.
#' @param FUN Function, which function should be used in summary;
#'   function should return single value per each level of by.
#' @param labelSum Character, label used for the overall breeding value.
#' @param subset Logical, perform summary only on a subset of
#'   \code{object} subsetted by this argument.
#' @param sums Logical, link between \code{\link[AlphaPart]{AlphaPart}}
#'   and \code{summary.AlphaPart()} (only for internal use!).
#' @param cov Logical, if FALSE returns \code{n} variances plus one
#'   additional column containing two times the sum of all covariances;
#'   otherwise returns \code{n} variance and \code{n(n-1)/2} covariances
#'   in the form of \code{2*Cov(., .)}, where \code{n} is the number of
#'   partitions. This argument only works when \code{FUN = var}. Defaut
#'   \code{cov = FALSE}.
#' @param ...  Arguments passed to other functions (not used at the
#'   moment).
#'
#' @example inst/examples//examples_summary.AlphaPart.R
#'
#' @return An object of class \code{summaryAlphaPart}, which is a list
#'   of data frames with summary statistics on breeding value
#'   partitions. For each trait there a dataframe holds summary for the
#'   "whole/original" breeding value and its partitions.  In addition
#'   another list is added (named \code{info}) with the following
#'   components holdinfg meta info:
#'
#'   * `path` column name holding path information
#'   * `nP` number of paths
#'   * `lP` path labels
#'   * `nT` number of traits
#'   * `lT` trait labels
#'   * `by` column name of variable by which summary was performed
#'   * `warn` potential warning messages associated with this object
#'   * `labelSum` column name of summary for "whole/original" breeding values
#'
#' There is a handy plot method (\code{\link[AlphaPart]{plot.summaryAlphaPart}}) for output.
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom dplyr group_by do
#' @importFrom magrittr %>%
#' @importFrom stats cov var
#'
#' @export

summary.AlphaPart <- function(
  object,
  by = NULL,
  FUN = mean,
  labelSum = "Sum",
  subset = NULL,
  sums = FALSE,
  cov = FALSE,
  ...
) {
  #---------------------------------------------------------------------
  ## --- Setup ---
  #---------------------------------------------------------------------
  groupSummary <- sums
  #---------------------------------------------------------------------
  # Test Alpha Part Class
  test <- !inherits(object, "AlphaPart")

  if (test) {
    stop("'object' must be of a AlphaPart class")
  }
  #---------------------------------------------------------------------
  # Transforming FUN in a function
  test <- is.character(FUN)
  if (test) {
    test <- is.function(try(get(FUN), silent = TRUE))
    if (!test) {
      stop("argument 'FUN' must be a function", call. = FALSE)
    }
    FUN <- get(FUN)
  }
  #---------------------------------------------------------------------
  # Test for covariance output
  test <- is.logical(cov)
  if (!test) {
    stop("argument 'cov' must be logical (TRUE or FALSE)", call. = FALSE)
  }
  #---------------------------------------------------------------------
  if (groupSummary) {
    by <- object$by
  }
  #---------------------------------------------------------------------
  if (!groupSummary) {
    test <- !is.null(by) && !(by %in% colnames(object[[1]]))
    if (test) {
      stop("argument 'by' must be NULL or one of column names in object")
    }
    test <- do.call(what = FUN, args = list(x = 1:3, ...))
    if (length(test) > 1) {
      stop("function FUN must return a single value (scalar)")
    }
  }
  #---------------------------------------------------------------------
  nC <- ncol(object[[1]]) ## number of columns
  nP <- object$info$nP ## number of paths
  nCov <- 0 ## number of covariances
  lP <- object$info$lP ## names  of paths
  nT <- object$info$nT ## number of traits
  lT <- object$info$lT ## names  of traits
  ret <- vector(mode = "list", length = nT + 1)
  names(ret) <- c(lT, "info")
  #---------------------------------------------------------------------
  ## Subset
  if (!is.null(subset)) {
    object[1:nT] <- lapply(object[1:nT], FUN = function(z) z[subset, ])
  }

  ret$info <- list(
    path = object$info$path,
    nP = nP,
    nCov = nCov,
    lP = lP,
    nT = nT,
    lT = lT,
    by = by,
    warn = object$info$warn,
    labelSum = labelSum
  )
  #---------------------------------------------------------------------
  ## --- Compute ---
  #---------------------------------------------------------------------
  z <- ifelse(groupSummary, 1, 2)
  for (i in 1:nT) {
    ## for each trait
    ## Setup
    cols <- c(lT[i], paste(lT[i], lP, sep = "_"))
    checkCov <- length(cols[-1]) > 1 ## do not run cov if path has 1 level
    paths <- cols
    paths[2:length(paths)] <- ret$info$lP
    paths[1] <- labelSum

    ## Summarize Variance Partitioning
    if (identical(deparse(FUN), deparse(var))) {
      . <- NULL
      if (!groupSummary) {
        if (is.null(by)) {
          #pri length ne sme biti na.rm = TRUE
          tmp <- rep(1, times = nrow(object[[i]]))
          tmpM <- aggregate(
            x = object[[i]][, cols],
            by = list(by = tmp),
            FUN = var,
            na.rm = TRUE
          )
          if (checkCov) {
            if (cov) {
              tmpM2 <- object[[i]] |>
                group_by(object[[i]][, by]) %>%
                do(data.frame(
                  cov = 2 *
                    t(cov(.[, cols[-1]], .[, cols[-1]])[
                      lower.tri(cov(.[, cols[-1]], .[, cols[-1]]), diag = FALSE)
                    ])
                ))
              if (!is.null(ncol(tmpM2))) {
                tmpM2 <- tmpM2[, -1]
              }
            } else {
              tmpM2 <- tmpM[, 2] - rowSums(tmpM[, -c(1:2)])
            }
            tmpM <- cbind(tmpM, tmpM2)
          }
          tmpN <- aggregate(
            x = object[[i]][, cols[1]],
            by = list(by = tmp),
            FUN = length
          )
        } else {
          tmpM <- aggregate(
            x = object[[i]][, cols],
            by = list(by = object[[i]][, by]),
            FUN = var,
            na.rm = TRUE
          )
          if (checkCov) {
            if (cov) {
              tmpM2 <- object[[i]] |>
                group_by(object[[i]][, by]) %>%
                do(data.frame(
                  cov = 2 *
                    t(cov(.[, cols[-1]], .[, cols[-1]])[
                      lower.tri(cov(.[, cols[-1]], .[, cols[-1]]), diag = FALSE)
                    ])
                ))
              if (!is.null(ncol(tmpM2))) {
                tmpM2 <- tmpM2[, -1]
              }
            } else {
              tmpM2 <- tmpM[, 2] - rowSums(tmpM[, -c(1:2)])
            }
            tmpM <- cbind(tmpM, tmpM2)
          }
          tmpN <- aggregate(
            x = object[[i]][, cols[1]],
            by = list(by = object[[i]][, by]),
            FUN = length
          )
        }
      } else {
        tmpN <- object$N[, c(1, i + 1)]
        tmpM <- object[[i]][, -(1:2)]
        tmpM <- cbind(rowSums(tmpM), tmpM)
        tmpM <- tmpM / tmpN[, 2]
      }
      #-----------------------------------------------------------------
      ## Add nice column names
      colnames(tmpN) <- c(by, "N")
      count <- seq(1, length(paths))[-1]
      if (cov && checkCov) {
        for (ii in count[-length(count)]) {
          for (jj in (ii + 1):max(count)) {
            kcount <- length(paths) + 1
            paths[kcount] <- paste0(paths[ii], paths[jj])
          }
        }
      } else {
        kcount <- length(paths) + 1
        if (checkCov) {
          paths <- c(paths, "Sum.Cov")
        }
      }
      colnames(tmpM)[z:ncol(tmpM)] <- paths
      #-----------------------------------------------------------------
      ## Combine FUN and number of records
      tmp <- cbind(tmpN, tmpM[, z:ncol(tmpM)])
      #-----------------------------------------------------------------
    } else {
      ## Summarize non-variance partitioning
      if (!groupSummary) {
        if (is.null(by)) {
          #pri length ne sme biti na.rm = TRUE
          tmp <- rep(1, times = nrow(object[[i]]))
          tmpM <- aggregate(
            x = object[[i]][, cols],
            by = list(by = tmp),
            FUN = FUN,
            na.rm = TRUE
          )
          tmpN <- aggregate(
            x = object[[i]][, cols[1]],
            by = list(by = tmp),
            FUN = length
          )
        } else {
          tmpM <- aggregate(
            x = object[[i]][, cols],
            by = list(by = object[[i]][, by]),
            FUN = FUN,
            na.rm = TRUE
          )
          tmpN <- aggregate(
            x = object[[i]][, cols[1]],
            by = list(by = object[[i]][, by]),
            FUN = length
          )
        }
      } else {
        tmpN <- object$N[, c(1, i + 1)]
        tmpM <- object[[i]][, -(1:2)]
        tmpM <- cbind(rowSums(tmpM), tmpM)
        tmpM <- tmpM / tmpN[, 2]
      }
      #-----------------------------------------------------------------
      ## Add nice column names
      colnames(tmpN) <- c(by, "N")
      colnames(tmpM)[z:ncol(tmpM)] <- paths
      #-----------------------------------------------------------------
      ## Combine FUN and number of records
      tmp <- cbind(tmpN, tmpM[, z:ncol(tmpM)])
      #-----------------------------------------------------------------
    }
    ## Store
    ret[[i]] <- tmp
  }
  #---------------------------------------------------------------------
  ## --- Update when var ---
  #---------------------------------------------------------------------
  if (identical(deparse(FUN), deparse(var))) {
    ret$info$nCov <- kcount - ret$info$nP - 1 # number of covariances
    ret$info$lP <- paths[-1] ## names of paths
  }
  #---------------------------------------------------------------------
  ## --- Return ---
  #---------------------------------------------------------------------
  class(ret) <- c("summaryAlphaPart", class(ret))
  ret
}

#=======================================================================
# Plot
# =======================================================================
#' @title A function to plot summary of partitioned breeding values.
#' @rdname plot.summaryAlphaPart
#' @method plot summaryAlphaPart
#' @usage \method{plot}{summaryAlphaPart}(x, by, sortValue,
#'   sortValueFUN, sortValueDec, addSum, paths, xlab, ylab, xlim, ylim,
#'   color, lineSize, lineType, lineTypeList, useDirectLabels, method,
#'   labelPath, ...)
#' @details Information in summaries of partitions of breeding values
#'   can be overhelming due to a large volume of numbers. Plot method
#'   can be used to visualise this data in eye pleasing way using
#'   ggplot2 graphics.
#'
#' @param x summaryAlphaPart, object from the \code{AlphaPart(...)} or
#'   \code{summary(AlphaPart(...), ...)} call.
#' @param by Character, the name of a column by which summary function
#'   FUN should be applied; if \code{NULL} (default) summary is given
#'   for the whole table.
#' @param sortValue Logical, affect legend attributes via sort of paths
#'   according to \code{sortValueFUN} function; if not logical, then
#'   ordered paths are given as a character vector.
#' @param sortValueFUN Function, that produces single value for one
#'   vector, say \code{mean} or \code{sum}.
#' @param sortValueDec Logical, sort decreasing.
#' @param addSum Logical, plot the overall trend.
#' @param paths Character or list or characters, name of paths to plot;
#'   if \code{NULL} plot all paths; see examples.
#' @param xlab Character, x-axis label.
#' @param ylab Character, y-axis label; can be a vector of several
#'   labels if there are more traits in \code{x} (recycled!).
#' @param xlim Numeric, a vector of two values with x-axis limits; use a
#'   list of vectors for more traits.
#' @param ylim Numeric, a vector of two values with y-axis limits; use a
#'   list of vectors for more traits.
#' @param color Character, color names; by default a set of 54 colors is
#'   predefined from the \pkg{RColorBrewer} package; in addition a black
#'   colour is attached at the begining for the overall trend; if there
#'   are more paths than colors then recycling occours.
#' @param lineSize Numeric, line width.
#' @param lineType Numeric, line type (recycled); can be used only if
#'   lineTypeList=NULL.
#' @param lineTypeList List, named list of numeric values that help to
#'   point out a set of paths (distinguished with line type) within
#'   upper level of paths (distinguished by, color), e.g.,
#'   lineTypeList=list("-1"=1, "-2"=2, def=1) will lead to use of line
#'   2, for paths having "-2" at the end of path name, while line type 1
#'   (default) will, be used for other paths; specification of this
#'   argument also causes recycling of colors for the upper level of
#'   paths; if NULL all lines have a standard line type, otherwise
#'   \code{lineType} does not have any effect.
#' @param useDirectLabels Logical, use directlabels package for legend.
#' @param method List, method for direct.label.
#' @param labelPath Character, legend title; used only if
#'   \code{useDirectLabels=FALSE}.
#' @param ...  Arguments passed to other functions (not used at the
#'   moment).
#'
#' @example inst/examples/examples_plotSummaryAlphaPart.R
#'
#' @return A list of ggplot objects that can be further modified or
#'   displayed.  For each trait in \code{x} there is one plot
#'   visualising summarized values.
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#'
#' @export
#' @importFrom directlabels last.qp
#' @importFrom directlabels direct.label
#' @importFrom reshape melt
#' @import ggplot2

plot.summaryAlphaPart <-
  function(
    x,
    by = NULL,
    sortValue = TRUE,
    sortValueFUN = sum,
    sortValueDec = TRUE,
    addSum = TRUE,
    paths = NULL,
    xlab = NULL,
    ylab = NULL,
    xlim = NULL,
    ylim = NULL,
    color,
    lineSize = 1,
    lineType = 1,
    lineTypeList = NULL,
    useDirectLabels = TRUE,
    method = list(last.qp, hjust = 0),
    labelPath = NULL,
    ...
  ) {
    #-------------------------------------------------------------------
    ## --- Setup ---
    #-------------------------------------------------------------------
    if (!inherits(x, "summaryAlphaPart")) {
      stop("'x' must be of a summaryAlphaPart class")
    }

    by <- x$info$by
    ## by argument
    if (is.null(by)) {
      stop(
        "output is provided only when the 'by' argument is defined on the 'summary' function"
      )
    }
    path <- x$info$path
    lT <- x$info$lT
    nT <- x$info$nT
    nP <- x$info$nP + x$info$nCov
    ret <- vector(mode = "list", length = nT)
    names(ret) <- x$info$lT

    ## Axis labels
    if (!is.null(xlab) && length(xlab) > 1) {
      stop("you can provide only one value for 'xlab'")
    }
    if (!is.null(ylab) && length(ylab) < nT) {
      ylab <- rep(ylab, length = nT)
    }

    ## Colors
    if (!missing(color)) {
      if (length(color) < nP) {
        color <- rep(color, length = nP)
      }
      color <- c("black", color)
    } else {
      if (FALSE) {
        ## Code to generate a bunch of qualitative colors
        requireNamespace("RColorBrewer")
        #library(package="RColorBrewer")
        pals <- c("Set1", "Dark2", "Accent", "Paired", "Set2", "Set3")
        palsN <- brewer.pal.info[pals, "maxcolors"]
        color <- vector(length = sum(palsN))
        j <- 1
        for (i in seq(along = pals)) {
          color[j:(j - 1 + palsN[i])] <-
            do.call("brewer.pal", args = list(n = palsN[i], name = pals[i]))
          j <- j + palsN[i]
        }
        color <- unique(color)
      }
      color <- c(
        "black",
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#FFFF33",
        "#A65628",
        "#F781BF",
        "#999999",
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
        "#A6761D",
        "#666666",
        "#7FC97F",
        "#BEAED4",
        "#FDC086",
        "#FFFF99",
        "#386CB0",
        "#F0027F",
        "#BF5B17",
        "#A6CEE3",
        "#1F78B4",
        "#B2DF8A",
        "#33A02C",
        "#FB9A99",
        "#E31A1C",
        "#FDBF6F",
        "#CAB2D6",
        "#6A3D9A",
        "#B15928",
        "#66C2A5",
        "#FC8D62",
        "#8DA0CB",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
        "#8DD3C7",
        "#FFFFB3",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        "#D9D9D9",
        "#BC80BD",
        "#CCEBC5",
        "#FFED6F"
      )
      color <- color[color != "#FFFF33"] ## remove yellow color(s)
    }
    #-------------------------------------------------------------------
    ## Line type
    if (is.null(lineTypeList)) {
      if (length(lineType) < nP) {
        lineType <- c(1, rep(x = lineType, times = nP))
      } else {
        lineType <- c(1, lineType)
      }
    }
    #-------------------------------------------------------------------
    ## --- Create plots ---
    #-------------------------------------------------------------------
    ## Ylim
    ylimT <- ylim

    for (i in 1:nT) {
      ## loop over traits
      colorI <- color
      lineTypeI <- lineType
      ## Prepare data
      tmp0 <- x[[i]]
      ## Make sure that path has a name "N"
      tmpCol <- colnames(tmp0)
      test <- tmpCol %in% "N"
      if (sum(test) > 1) {
        tmpCol[test & cumsum(test) == 2] <- "N."
        colnames(tmp0) <- tmpCol
        warning("changing path name from 'N' to 'N.'")
      }
      tmp <- melt(tmp0[, !(colnames(tmp0) %in% "N")], id = by)
      colnames(tmp) <- c("by", "path", "trait")
      if (is.logical(sortValue)) {
        if (sortValue) {
          nC <- ncol(tmp0)
          pathStat <- sapply(
            X = tmp0[, (nC - nP + 1):nC],
            FUN = sortValueFUN,
            na.rm = TRUE
          )
          levs <- names(sort(pathStat, decreasing = sortValueDec))
          tmp$path <- factor(tmp$path, levels = c(x$info$labelSum, levs))
          if (!is.null(lineTypeList)) {
            ## fiddle with upper (color) and lower (line type) level of paths
            levs2X <- names(lineTypeList)
            levs2X <-
              levs2X[levs2X != "def"]
            levs1 <- levs2 <- levs
            for (k in levs2X) {
              j <- paste(k, "$", sep = "") ## lower label mark needs to be at the end of path name!!!
              levs2[grep(pattern = j, x = levs1)] <- k
              levs1 <- sub(pattern = j, replacement = "", x = levs1)
            }
            levs2[!levs2 %in% levs2X] <- "def"
            levs1X <- unique(levs1)
            colorList <- as.list(color[-1][seq_len(length(levs1X))])
            names(colorList) <- levs1X
            colorI <- c("black", unlist(colorList[levs1]))
            names(colorI) <- NULL
            lineTypeI <- c(lineTypeList$def, unlist(lineTypeList[levs2]))
            names(lineTypeI) <- NULL
          }
        }
      } else {
        tmp$path <- factor(tmp$path, levels = c(x$info$labelSum, sortValue))
      }
      #-----------------------------------------------------------------
      ## Prepare plot
      #trait in "" since it is not defined
      #-----------------------------------------------------------------
      trait <- tmp$trait
      p <- qplot(
        x = by,
        y = trait,
        group = path,
        data = tmp,
        color = path,
        linetype = path,
        geom = "line"
      )
      p <- p + geom_line(size = lineSize)
      p <- p + xlab(label = ifelse(is.null(xlab), by, xlab))
      p <- p + ylab(label = ifelse(is.null(ylab), lT[i], ylab[i])) #lT[i] is the TRAIT!!!
      #-----------------------------------------------------------------
      if (!is.null(xlim)) {
        if (is.list(xlim)) {
          xlimI <- xlim[[i]]
        } else {
          xlimI <- xlim
        }
        p <- p + scale_x_continuous(limits = xlimI)
      }
      #-----------------------------------------------------------------
      if (!is.null(ylimT)) {
        if (is.list(ylimT)) {
          ylimI <- ylimT[[i]]
        } else {
          ylimI <- ylimT
        }
        p <- p + scale_y_continuous(limits = ylimI)
      }
      #-----------------------------------------------------------------
      if (useDirectLabels) {
        p <- directlabels::direct.label(p = p, method = method)
      }
      #-----------------------------------------------------------------
      ## This needs to follow direct.label
      #-----------------------------------------------------------------
      p <- p +
        scale_colour_manual(
          values = colorI,
          name = ifelse(is.null(labelPath), path, labelPath)
        )
      p <- p +
        scale_linetype_manual(
          values = lineTypeI,
          name = ifelse(is.null(labelPath), path, labelPath)
        )
      ret[[i]] <- p
    }
    #-------------------------------------------------------------------
    ## --- Return ---
    #-------------------------------------------------------------------
    class(ret) <- c("plotSummaryAlphaPart", class(ret))
    ret
  }

#=======================================================================
# save
# ======================================================================

#' @title Save plot method for \code{AlphaPart}
#' @return Beside the side effect of saving plots to disk, filenames are printed on
#' screen during the process and at the end invisibly returned.
#' @param ... Arguments passed to \code{type} specific methods, say
#' \code{width} and \code{height} for \code{type="pdf"} etc.
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom grDevices dev.cur
#' @importFrom grDevices dev.off
#' @export

savePlot <- function(...) {
  UseMethod("savePlot")
}

#' @title Save plot objects on the disk for permanent storage. Function
#'   \code{\link[grDevices]{savePlot}} from the \pkg{grDevices} package
#'   works for current page on graphical device. This is an attempt to
#'   make this function generic so that one can define \code{savePlot}
#'   methods for particular needs.
#' @rdname savePlot.plotSummaryAlphaPart
#' @method savePlot plotSummaryAlphaPart
#' @usage \method{savePlot}{plotSummaryAlphaPart}(x, filename, type,
#'   device, pre.hook, traitsAsDir, ...)
#' @aliases savePlot.plotSummaryAlphaPart
#'
#' @description Save plot objects of class
#'   \code{plotSummaryAlphaPart} on the disk for permanent storage.
#'
#' @seealso \code{\link[grDevices]{savePlot}} help page on the default \code{savePlot}
#' method in the \pkg{grDevices} package; \code{\link[AlphaPart]{savePlot.plotSummaryAlphaPart}}
#' help page on the method for the objects of \code{plotSummaryAlphaPart} class; and
#' \code{\link[AlphaPart]{plot.summaryAlphaPart}} for ploting results of summaryAlphaPart object.
#'
#' @param x Object on which to chose savePLot method.
#' @param filename Character, filename to save to.
#' @param type  Character, file/device type.
#' @param device Device, the device to save from.
#' @param pre.hook Function, call some code before calling print method for plots (see examples).
#' @param traitsAsDir Logical, should plots be saved within trait folders; the construction is
#' \code{file.path(dirname(file), trait, basename(file))}.
#' folders are created if they do not exist.
#' @param ... Arguments passed to \code{type} specific methods, say
#' \code{width} and \code{height} for \code{type="pdf"} etc.
#'
#' @example /inst/examples/examples_savePlot.plotsummaryAlphaPart.R
#'
#' @return Beside the side effect of saving plots to disk, filenames are printed on
#' screen during the process and at the end invisibly returned.
#'
#' @useDynLib AlphaPart, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom grDevices dev.cur
#' @importFrom grDevices dev.off
#' @export

savePlot.plotSummaryAlphaPart <- function(
  x, ##<< plotSummaryAlphaPart, output object from
  ## \code{\link[AlphaPart]{plot.summaryAlphaPart}} function
  filename = paste("Rplot", type, sep = "."), ##<< character, filename to save to
  type = c("pdf", "png", "jpeg", "tiff", "bmp"), ##<< character, file/device type
  device = dev.cur(), ##<< device, the device to save from (not used for this method)
  pre.hook = NULL, ##<<
  traitsAsDir = FALSE, ##<<
  ... ##<<
) {
  if (length(filename) > 1) {
    stop("'filename' argument must be of length one")
  }
  if (!inherits(x, "plotSummaryAlphaPart")) {
    stop("'x' must be of a 'plotSummaryAlphaPart' class")
  }
  filenameOrig <- sub(
    pattern = paste(".", type, "$", sep = ""),
    replacement = "",
    x = filename
  )
  ret <- NULL
  lT <- names(x)

  for (i in seq_len(length(x))) {
    if (traitsAsDir) {
      dir.create(
        path = file.path(dirname(filenameOrig), lT[i]),
        recursive = TRUE,
        showWarnings = FALSE
      )
      filename <- file.path(
        dirname(filenameOrig),
        lT[i],
        basename(filenameOrig)
      )
    }
    fileA <- paste(filename, paste(lT[i], ".", type, sep = ""), sep = "_")
    ret <- c(ret, fileA)
    cat(fileA, "\n")
    print(list(file = fileA, ...))
    do.call(what = type, args = list(file = fileA, ...))
    if (!is.null(pre.hook) && is.function(pre.hook)) {
      pre.hook()
    }
    print(x[[i]])
    dev.off()
  }
  #---------------------------------------------------------------------
  ## --- Return ---
  #---------------------------------------------------------------------
  invisible(ret)
}

#' @rdname savePlot.plotSummaryAlphaPart
#' @method savePlot default
#' @aliases savePlot.default
#' @export

savePlot.default <- function(...) {
  ##seealso<< \code{\link[grDevices]{savePlot}} help page on the default
  ## \code{savePlot} method in the \pkg{grDevices} package

  grDevices::savePlot(...)

  ##value<< See \code{\link[grDevices]{savePlot}} for details.
}

#=======================================================================
# center base population
#=======================================================================
#' @title Calculate parent average for base population.
#' @description This is an internally called functions used to calculate
#' parent average for base population.
#'
#' @usage NULL
#'
#' @seealso
#' \code{\link[AlphaPart]{AlphaPart}}
#'
#' @author Thiago de Paula Oliveira
#'
#' @keywords internal
#' @importFrom stats lm confint
#' @export

centerPop <- function(y, colBV, path) {
  #---------------------------------------------------------------------
  # Selecting founders and missing pedigree animals
  #---------------------------------------------------------------------
  colBV <- (ncol(y) - length(colBV) + 1):ncol(y)
  if (length(colBV) == 1) {
    tmp <- as.matrix(y[c(y[, 2] == 0 & y[, 3] == 0), colBV])
  } else {
    tmp <- y[c(y[, 2] == 0 & y[, 3] == 0), colBV]
  }
  baseMean <- colMeans(tmp, na.rm = TRUE)
  #---------------------------------------------------------------------
  # Decision criteria
  #---------------------------------------------------------------------
  basePop <- apply(y[, c(2, 3)] == 0, 1, all)
  for (i in seq_len(ncol(tmp))) {
    if (all(confint(lm(tmp[, i] ~ 1), level = 0.95) > 0)) {
      path$w[-1, i] <- path$w[-1, i] - basePop * baseMean[i]
      path$pa[-1, i] <- path$pa[-1, i] +
        basePop * y[, colBV[i]] -
        path$w[-1, i] * basePop
    }
  }
  return(path)
}

#=======================================================================
# Scaling EBVs
#=======================================================================
#' @title Scale EBVs for objects of the class summaryAlphaPart.
#' @description   This is an internally called functions used to Scale
#' EBVs in respect to base population for objects of the class
#' \code{AlphaPart}.
#'
#' @usage NULL
#'
#' @seealso
#' \code{\link[AlphaPart]{AlphaPart}}
#'
#' @author Thiago de Paula Oliveira
#'
#' @keywords internal
#' @importFrom stats sd
#' @export

sEBV <- function(y, center, scale, recode, unknown) {
  id. <- 1
  fid. <- 2
  mid. <- 3
  if (recode) {
    y[, id.:mid.] <- cbind(
      id = seq_len(nrow(y)),
      fid = match(y[, fid.], y[, id.], nomatch = 0),
      mid = match(y[, mid.], y[, id.], nomatch = 0)
    )
  } else {
    ## Make sure we have 0 when recoded data is provided
    if (is.na(unknown)) {
      y[, c(fid., mid.)] <- NAToUnknown(x = y[, c(fid., mid.)], unknown = 0)
    } else {
      if (unknown != 0) {
        y[, c(fid., mid.)] <-
          NAToUnknown(
            x = unknownToNA(x = y[, c(fid., mid.)], unknown = unknown),
            unknown = 0
          )
      }
    }
  }
  #---------------------------------------------------------------------
  # Selecting founders and missing pedigree animals
  #---------------------------------------------------------------------
  if (ncol(y) == (mid. + 1)) {
    tmp <- as.matrix(y[c(y[, fid.] == 0 & y[, mid.] == 0), -c(id.:mid.)])
    y <- as.matrix(y[, -c(id.:mid.)])
  } else {
    tmp <- y[c(y[, fid.] == 0 & y[, mid.] == 0), -c(id.:mid.)]
    y <- y[, -c(id.:mid.)]
  }
  #---------------------------------------------------------------------
  # Centering
  #---------------------------------------------------------------------
  if (is.logical(center)) {
    if (center) {
      center <- colMeans(tmp, na.rm = TRUE)
      y <- y -
        rep(center, rep.int(nrow(y), ncol(y)))
    }
  }
  #---------------------------------------------------------------------
  # Scaling
  #---------------------------------------------------------------------
  if (is.logical(scale)) {
    if (scale) {
      f <- function(x) {
        sd(x, na.rm = TRUE)
      }
      scale <- apply(tmp, 2L, f)
      y <- y /
        rep(scale, rep.int(nrow(y), ncol(y)))
    }
  }
  return(y)
}

#' @title Get scale information
#' @description   This is an internally called function
#'
#' @usage NULL
#'
#' @seealso
#' \code{\link[AlphaPart]{AlphaPart}}
#'
#' @author Thiago de Paula Oliveira
#'
#' @keywords internal
#' @export
getScale <- function(center = FALSE, scale = FALSE, ...) {
  list(center = center, scale = scale, ...)
}

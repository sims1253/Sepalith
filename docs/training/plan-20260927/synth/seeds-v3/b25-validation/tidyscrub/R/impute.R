#' Carry values forward or backward over gaps
#'
#' @param x A vector with missing values.
#' @param max.gap Longest run of missing values to fill; longer runs are
#'   left as they are.
#' @return `x` with gaps filled from the previous (`fill.forward()`) or the
#'   next (`fill.backward()`) observed value.
#' @rdname fill
#' @export
fill.forward <- function(x, max.gap = Inf) {
  observed <- which(!is.na(x))
  if (length(observed) == 0L) {
    return(x)
  }
  index <- findInterval(seq_along(x), observed)
  source <- rep(NA_integer_, length(x))
  source[index > 0L] <- observed[index]
  gap <- seq_along(x) - source
  fill <- is.na(x) & !is.na(source) & gap <= max.gap
  x[fill] <- x[source[fill]]
  x
}

#' @rdname fill
#' @export
fill.backward <- function(x, max.gap = Inf) {
  rev(fill.forward(rev(x), max.gap = max.gap))
}

#' Most frequent value
#'
#' @param x A vector.
#' @return The most frequent non-missing value of `x`; ties go to the value
#'   seen first. `NA` when `x` has no observed values.
#' @keywords internal
most.frequent <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(NA)
  }
  values <- unique(x)
  counts <- tabulate(match(x, values))
  values[which.max(counts)]
}

#' Impute missing values in one column
#'
#' @param x A column vector.
#' @param method One of `"mean"`, `"median"`, `"mode"`, `"constant"`,
#'   `"locf"`, `"nocb"` or `"linear"`.
#' @param value Replacement used by `"constant"`.
#' @return `x` with missing values replaced; the attribute `"imputed"`
#'   holds the positions that were filled.
#' @export
impute.column <- function(x,
                          method = c("mean", "median", "mode", "constant", "locf", "nocb",
                                     "linear"),
                          value = NULL) {
  method <- match.arg(method)
  was.na <- is.na(x)
  out <- switch(method,
    mean = {
      if (!is.numeric(x)) stop("`method = \"mean\"` needs a numeric column.", call. = FALSE)
      replace(x, was.na, mean(x, na.rm = TRUE))
    },
    median = {
      if (!is.numeric(x)) stop("`method = \"median\"` needs a numeric column.", call. = FALSE)
      replace(x, was.na, stats::median(x, na.rm = TRUE))
    },
    mode = {
      replace(x, was.na, most.frequent(x))
    },
    constant = {
      if (is.null(value)) stop("`method = \"constant\"` needs a `value`.", call. = FALSE)
      replace(x, was.na, value)
    },
    locf = {
      fill.forward(x)
    },
    nocb = {
      fill.backward(x)
    },
    linear = {
      if (!is.numeric(x)) stop("`method = \"linear\"` needs a numeric column.", call. = FALSE)
      observed <- which(!was.na)
      stats::approx(observed, x[observed], xout = seq_along(x), rule = 1)$y
    }
  )
  attr(out, "imputed") <- which(was.na & !is.na(out))
  out
}

#' Impute missing values in several columns
#'
#' @param data A data frame to clean.
#' @param methods Named character vector mapping column names to
#'   [impute.column()] methods.
#' @param flag Whether to add a logical column `<name>.imputed` per imputed
#'   column.
#' @return A list with elements `data` and `n.imputed` (a named integer
#'   vector with one count per imputed column).
#' @export
impute.frame <- function(data, methods, flag = FALSE) {
  unknown <- setdiff(names(methods), names(data))
  if (length(unknown) > 0L) {
    stop("`methods` names unknown column(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }
  n.imputed <- integer()
  for (col in names(methods)) {
    filled <- impute.column(data[[col]], method = methods[[col]])
    positions <- attr(filled, "imputed")
    attr(filled, "imputed") <- NULL
    data[[col]] <- filled
    n.imputed[[col]] <- length(positions)
    if (flag) {
      data[[paste0(col, ".imputed")]] <- seq_len(nrow(data)) %in% positions
    }
  }
  list(data = data, n.imputed = n.imputed)
}

#' Impute within groups
#'
#' @param data A data frame to clean.
#' @param column Column to impute.
#' @param group Column whose values define the groups.
#' @param method Method passed to [impute.column()].
#' @return `data` with `column` imputed separately within each group.
#' @export
impute.by.group <- function(data, column, group, method = "median") {
  unknown <- setdiff(c(column, group), names(data))
  if (length(unknown) > 0L) {
    stop("`data` lacks column(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  x <- data[[column]]
  for (rows in split(seq_len(nrow(data)), data[[group]])) {
    if (all(is.na(x[rows]))) {
      next
    }
    filled <- impute.column(x[rows], method = method)
    x[rows] <- as.vector(filled)
  }
  data[[column]] <- x
  data
}

#' Suggest an imputation method per column
#'
#' @description
#' Picks an [impute.column()] method for every column with missing values,
#' based on the column type: `"median"` for skewed numeric columns and
#' `"mean"` otherwise, `"mode"` for character, factor and logical columns,
#' and `"locf"` for dates.
#'
#' @param data A data frame to clean.
#' @param skew.cutoff Absolute skewness above which a numeric column counts
#'   as skewed.
#' @return A named character vector of methods, one per column that has
#'   missing values, suitable for [impute.frame()].
#' @export
suggest.imputation <- function(data, skew.cutoff = 1) {
  methods <- character()
  for (col in names(data)) {
    x <- data[[col]]
    if (!anyNA(x) || all(is.na(x))) {
      next
    }
    kind <- if (inherits(x, "Date")) "date" else class(x)[1]
    methods[[col]] <- switch(kind,
      numeric = ,
      integer = {
        obs <- x[!is.na(x)]
        spread <- stats::sd(obs)
        skew <- if (is.na(spread) || spread == 0) 0 else mean((obs - mean(obs))^3) / spread^3
        if (abs(skew) > skew.cutoff) "median" else "mean"
      },
      character = {
        "mode"
      },
      factor = {
        "mode"
      },
      logical = {
        "mode"
      },
      date = {
        "locf"
      },
      "constant"
    )
  }
  methods
}

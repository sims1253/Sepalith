#---------------------------------------------------------------
# Name:        swash (swash package)
# Purpose:     Health Geography Toolbox for Model-Based
#              Analysis of Infections Panel Data
# Author:      Thomas Wieland
#              ORCID: 0000-0001-5168-9846
#              mail: geowieland@googlemail.com
# Version:     2.0.0
# Last update: 2026-04-05 22:03
# Copyright (c) 2020-2026 Thomas Wieland
#---------------------------------------------------------------

library(sf)
library(spdep)
library(zoo)
library(strucchange)
library(lubridate)


package_name <- "swash"
package_version <- "2.0.0"


permitted_other_cols <- c(
  "R_t",
  # Effective reproduction number
  "Cum. cases",
  # Cumulative cases
  "Incidence",
  # Incidence (per xxx pop)
  "Population",
  # Population size of the region
  "Roll. mean",
  # Rolling mean of cases
  "Roll. sum"
  # Rolling sum of cases
)

model_descriptions <- list(
  "sbm" = "Swash-Backwash Model for the Single Epidemic Wave",
  "loggrowth" = "Logistic Growth Model",
  "expgrowth" = "Exponential Growth Model",
  "hawkes" = "Hawkes Process",
  "breaksgrowth" = "Time Series Model with Breakpoints"
)


# Class infpan:
setClass(
  "infpan",
  slots = list(
    input_data = "data.frame",
    data_statistics = "numeric",
    index_col_names = "character",
    cases_col_name = "character",
    other_cols = "character",
    time_format = "character",
    time_unit = "character",
    timestamp = "list"
  ),
  prototype = list(
    other_cols = character(0),
    time_format = "%Y-%m-%d",
    time_unit = "days",
    timestamp = list()
  )
)

# Methods of class infpan:
setMethod(
  "summary",
  "infpan",
  function(object) {
    cat("Infections panel data\n\n")

    cat("Input data\n")
    cat(sprintf("  Units       : %s\n", object@data_statistics[1]))
    cat(sprintf(
      "  Time points : %s\n",
      paste0(object@data_statistics[2], " ", object@time_unit)
    ))
    cat(sprintf(
      "  Balanced    : %s\n",
      ifelse(object@data_statistics[4], "YES", "NO")
    ))

    cat("Main columns\n")
    cat(sprintf("  Units       : %s\n", object@index_col_names[1]))
    cat(sprintf("  Time points : %s\n", object@index_col_names[2]))
    cat(sprintf("  Cases       : %s\n", object@cases_col_name))

    other_cols <- object@other_cols

    if (length(other_cols) > 0) {
      cat("Other columns\n")

      for (name in names(other_cols)) {
        if (name %in% permitted_other_cols) {
          cat(sprintf("  %-11s : %s\n", name, other_cols[[name]]))
        }
      }
    }

    invisible(object)
  }
)

setMethod(
  "print",
  "infpan",
  function(x) {
    cat(
      paste0(
        "Infections panel data with ",
        x@data_statistics[1],
        " spatial units and ",
        x@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for details")

    invisible(x)
  }
)

setMethod(
  "show",
  "infpan",
  function(object) {
    cat(
      paste0(
        "Infections panel data with ",
        object@data_statistics[1],
        " spatial units and ",
        object@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for details")

    invisible(object)
  }
)

setGeneric(
  "calculate_Rt",
  function(
    object,
    GP = 4,
    correction = FALSE,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("calculate_Rt")
  }
)

setMethod(
  "calculate_Rt",
  "infpan",
  function(
    object,
    GP = 4,
    correction = FALSE,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    input_data <- object@input_data

    other_cols <- object@other_cols

    if ("R_t" %in% names(other_cols)) {
      col_R_t <- other_cols[["R_t"]]

      if (isFALSE(overwrite)) {
        warning(
          paste0(
            "Effective reproduction number already included in column '",
            col_R_t,
            "' and overwrite is set to FALSE"
          ),
          "\n"
        )

        return(invisible(object))
      } else {
        if (isTRUE(verbose)) {
          message(
            paste0(
              "Effective reproduction number included in column '",
              col_R_t,
              "' will be overwritten"
            ),
            "\n"
          )
        }

        input_data[col_R_t] <- NULL
      }
    }

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]
    col_cases <- object@cases_col_name

    N <- object@data_statistics[1]
    N_names <- levels(as.factor(input_data[[col_region]]))

    if (is.null(col_name)) {
      col_name <- "R_t"
    }

    data_Rt <-
      data.frame(
        matrix(ncol = 2)
      )
    colnames(data_Rt) <-
      c(
        paste0(col_region, "_", col_date),
        col_name
      )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating effective reproduction number from cases in column '",
        col_cases,
        "' with GI=",
        GP,
        " for ",
        N,
        " regions in column '",
        col_name,
        "' ... "
      ))
    }

    i <- 0

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <- input_data_i[
        order(input_data_i[[col_region]], input_data_i[[col_date]]),
      ]

      input_data_i_Rt <-
        R_t(
          infections = input_data_i[[col_cases]],
          GP = GP,
          correction = correction
        )

      data_Rt_i <- data.frame(
        input_data_i[[paste0(col_region, "_", col_date)]],
        input_data_i_Rt[1]
      )
      colnames(data_Rt_i) <- colnames(data_Rt)

      data_Rt <- rbind(
        data_Rt,
        data_Rt_i
      )
    }

    data_Rt <- data_Rt[!is.na(data_Rt[[paste0(col_region, "_", col_date)]]), ]

    input_data_Rt <-
      merge(
        input_data,
        data_Rt,
        by.x = paste0(col_region, "_", col_date),
        by.y = paste0(col_region, "_", col_date)
      )

    other_cols[["R_t"]] <- col_name

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(input_data_Rt),
        data_statistics = object@data_statistics,
        index_col_names = object@index_col_names,
        cases_col_name = object@cases_col_name,
        other_cols = other_cols,
        timestamp = object@timestamp
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "calculate_Rt(infpan)",
      process = paste0(
        "Calculated effective reproduction number from cases in column '",
        col_cases,
        "' with GI=",
        GP,
        " for ",
        N,
        " regions in column '",
        col_name,
        "'"
      )
    )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    invisible(infpan_object)
  }
)

setGeneric(
  "calculate_rollmean",
  function(
    object,
    k = 7,
    align = "center",
    fill = NA,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("calculate_rollmean")
  }
)

setMethod(
  "calculate_rollmean",
  "infpan",
  function(
    object,
    k = 7,
    align = "center",
    fill = NA,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    input_data <- object@input_data

    other_cols <- object@other_cols

    if (permitted_other_cols[5] %in% names(other_cols)) {
      col_rollmean <- other_cols[[permitted_other_cols[5]]]

      if (isFALSE(overwrite)) {
        warning(
          paste0(
            "Rolling mean already included in column '",
            col_rollmean,
            "' and overwrite is set to FALSE"
          ),
          "\n"
        )

        return(invisible(object))
      } else {
        if (isTRUE(verbose)) {
          message(
            paste0(
              "Rolling mean included in column '",
              col_rollmean,
              "' will be overwritten"
            ),
            "\n"
          )
        }

        input_data[col_rollmean] <- NULL
      }
    }

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]
    col_cases <- object@cases_col_name

    N <- object@data_statistics[1]
    N_names <- levels(as.factor(input_data[[col_region]]))

    if (is.null(col_name)) {
      col_name <- paste0(col_cases, "_rm")
    }

    data_rollmean <-
      data.frame(
        matrix(ncol = 2)
      )
    colnames(data_rollmean) <-
      c(
        paste0(col_region, "_", col_date),
        col_name
      )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating rolling mean from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions with k=",
        k,
        " and fill=",
        fill,
        " in column '",
        col_name,
        "' ... "
      ))
    }

    i <- 0

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <- input_data_i[
        order(input_data_i[[col_region]], input_data_i[[col_date]]),
      ]

      input_data_i_rollmean <-
        rollmean(
          input_data_i[[col_cases]],
          k = k,
          fill = fill,
          align = align
        )

      data_rollmean_i <- data.frame(
        input_data_i[[paste0(col_region, "_", col_date)]],
        input_data_i_rollmean
      )
      colnames(data_rollmean_i) <- colnames(data_rollmean)

      data_rollmean <- rbind(
        data_rollmean,
        data_rollmean_i
      )
    }

    data_rollmean <- data_rollmean[
      !is.na(data_rollmean[[paste0(col_region, "_", col_date)]]),
    ]

    input_data_rollmean <-
      merge(
        input_data,
        data_rollmean,
        by.x = paste0(col_region, "_", col_date),
        by.y = paste0(col_region, "_", col_date)
      )

    other_cols <- object@other_cols
    other_cols[[permitted_other_cols[5]]] <- col_name

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(input_data_rollmean),
        data_statistics = object@data_statistics,
        index_col_names = object@index_col_names,
        cases_col_name = object@cases_col_name,
        other_cols = other_cols,
        timestamp = object@timestamp
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "calculate_rollmean(infpan)",
      process = paste0(
        "Calculated rolling mean from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions with k=",
        k,
        " and fill=",
        fill,
        " in column '",
        col_name,
        "'"
      )
    )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    invisible(infpan_object)
  }
)

setGeneric(
  "calculate_rollsum",
  function(
    object,
    k = 7,
    align = "center",
    fill = NA,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("calculate_rollsum")
  }
)

setMethod(
  "calculate_rollsum",
  "infpan",
  function(
    object,
    k = 7,
    align = "center",
    fill = NA,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    input_data <- object@input_data

    other_cols <- object@other_cols

    if (permitted_other_cols[6] %in% names(other_cols)) {
      col_rollsum <- other_cols[[permitted_other_cols[6]]]

      if (isFALSE(overwrite)) {
        warning(
          paste0(
            "Rolling sum already included in column '",
            col_rollsum,
            "' and overwrite is set to FALSE"
          ),
          "\n"
        )

        return(invisible(object))
      } else {
        if (isTRUE(verbose)) {
          message(
            paste0(
              "Rolling sum included in column '",
              col_rollsum,
              "' will be overwritten"
            ),
            "\n"
          )
        }

        input_data[col_rollsum] <- NULL
      }
    }

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]
    col_cases <- object@cases_col_name

    N <- object@data_statistics[1]
    N_names <- levels(as.factor(input_data[[col_region]]))

    if (is.null(col_name)) {
      col_name <- paste0(col_cases, "_rs")
    }

    data_rollsum <-
      data.frame(
        matrix(ncol = 2)
      )
    colnames(data_rollsum) <-
      c(
        paste0(col_region, "_", col_date),
        col_name
      )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating rolling sum from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions with k=",
        k,
        " and fill=",
        fill,
        " in column '",
        col_name,
        "' ... "
      ))
    }

    i <- 0

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <- input_data_i[
        order(input_data_i[[col_region]], input_data_i[[col_date]]),
      ]

      input_data_i_rollsum <-
        rollsum(
          input_data_i[[col_cases]],
          k = k,
          fill = fill,
          align = align
        )

      data_rollsum_i <- data.frame(
        input_data_i[[paste0(col_region, "_", col_date)]],
        input_data_i_rollsum
      )
      colnames(data_rollsum_i) <- colnames(data_rollsum)

      data_rollsum <- rbind(
        data_rollsum,
        data_rollsum_i
      )
    }

    data_rollsum <- data_rollsum[
      !is.na(data_rollsum[[paste0(col_region, "_", col_date)]]),
    ]

    input_data_rollsum <-
      merge(
        input_data,
        data_rollsum,
        by.x = paste0(col_region, "_", col_date),
        by.y = paste0(col_region, "_", col_date)
      )

    other_cols <- object@other_cols
    other_cols[[permitted_other_cols[6]]] <- col_name

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(input_data_rollsum),
        data_statistics = object@data_statistics,
        index_col_names = object@index_col_names,
        cases_col_name = object@cases_col_name,
        other_cols = other_cols,
        timestamp = object@timestamp
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "calculate_rollsum(infpan)",
      process = paste0(
        "Calculated rolling sum from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions with k=",
        k,
        " and fill=",
        fill,
        " in column '",
        col_name,
        "'"
      )
    )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    invisible(infpan_object)
  }
)

setGeneric(
  "calculate_cum",
  function(
    object,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("calculate_cum")
  }
)

setMethod(
  "calculate_cum",
  "infpan",
  function(
    object,
    col_name = NULL,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    input_data <- object@input_data

    other_cols <- object@other_cols

    if ("Cum. cases" %in% names(other_cols)) {
      col_cum_cases <- other_cols[["Cum. cases"]]

      if (isFALSE(overwrite)) {
        warning(
          paste0(
            "Cumulative infections already included in column '",
            col_cum_cases,
            "' and overwrite is set to FALSE"
          ),
          "\n"
        )

        return(invisible(object))
      } else {
        message(
          paste0(
            "Cumulative infections included in column '",
            col_cum_cases,
            "' will be overwritten"
          ),
          "\n"
        )

        input_data[col_cum_cases] <- NULL
      }
    }

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]
    col_cases <- object@cases_col_name

    N <- object@data_statistics[1]
    N_names <- levels(as.factor(input_data[[col_region]]))

    if (is.null(col_name)) {
      col_name <- paste0(col_cases, "_cum")
    }

    data_cum <-
      data.frame(
        matrix(ncol = 2)
      )
    colnames(data_cum) <-
      c(
        paste0(col_region, "_", col_date),
        col_name
      )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating cumulative infections from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions in column '",
        col_name,
        "' ... "
      ))
    }

    i <- 0

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <- input_data_i[
        order(input_data_i[[col_region]], input_data_i[[col_date]]),
      ]

      input_data_i_cum <-
        data.frame(
          cumsum(input_data_i[[col_cases]]),
          input_data_i[[col_date]]
        )
      colnames(input_data_i_cum) <- c("y", "t")

      data_cum_i <- data.frame(
        input_data_i[[paste0(col_region, "_", col_date)]],
        input_data_i_cum$y
      )
      colnames(data_cum_i) <- colnames(data_cum)

      data_cum <- rbind(
        data_cum,
        data_cum_i
      )
    }

    data_cum <- data_cum[
      !is.na(data_cum[[paste0(col_region, "_", col_date)]]),
    ]

    input_data_cum <-
      merge(
        input_data,
        data_cum,
        by.x = paste0(col_region, "_", col_date),
        by.y = paste0(col_region, "_", col_date)
      )

    other_cols <- object@other_cols
    other_cols[["Cum. cases"]] <- col_name

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(input_data_cum),
        data_statistics = object@data_statistics,
        index_col_names = object@index_col_names,
        cases_col_name = object@cases_col_name,
        other_cols = other_cols,
        timestamp = object@timestamp
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "calculate_cum(infpan)",
      process = paste0(
        "Calculated cumulative infections from cases in column '",
        col_cases,
        "' for ",
        N,
        " regions in column '",
        col_name,
        "'"
      )
    )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    invisible(infpan_object)
  }
)

setGeneric(
  "calculate_incidence",
  function(
    object,
    use_column = NULL,
    col_name = NULL,
    pop_factor = 100000,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("calculate_incidence")
  }
)

setMethod(
  "calculate_incidence",
  "infpan",
  function(
    object,
    use_column = "Cases",
    col_name = NULL,
    pop_factor = 100000,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    input_data <- object@input_data

    col_cases <- object@cases_col_name
    other_cols <- object@other_cols

    if (!permitted_other_cols[4] %in% names(other_cols)) {
      stop(
        "No population column defined. Calculation of incidence is not possible."
      )
    }

    if (permitted_other_cols[3] %in% names(other_cols)) {
      col_incidence <- other_cols[[permitted_other_cols[3]]]

      if (isFALSE(overwrite)) {
        warning(
          paste0(
            "Incidence already included in column '",
            col_incidence,
            "' and overwrite is set to FALSE"
          ),
          "\n"
        )

        return(invisible(object))
      } else {
        if (isTRUE(verbose)) {
          message(
            paste0(
              "Incidence included in column '",
              col_incidence,
              "' will be overwritten"
            ),
            "\n"
          )
        }

        input_data[col_incidence] <- NULL
      }
    }

    col_numerator <- col_cases

    if (use_column != "Cases") {
      permitted_other_numerator_cols <- permitted_other_cols[c(2, 5:6)]

      if (use_column %in% permitted_other_numerator_cols) {
        col_numerator <- other_cols[[use_column]]
      } else {
        warning(
          paste0(
            "Column identifier '",
            use_column,
            "' is unknown. Permitted identifier for other columns in incidence calculation are: ",
            paste(permitted_other_numerator_cols, collapse = ", "),
            "."
          ),
          "\n"
        )
      }
    }

    col_denominator <- other_cols[[permitted_other_cols[4]]]

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    N <- object@data_statistics[1]
    N_names <- levels(as.factor(input_data[[col_region]]))

    if (is.null(col_name)) {
      col_name <- paste0(col_cases, "_inc")
    }

    data_incidence <-
      data.frame(
        matrix(ncol = 2)
      )
    colnames(data_incidence) <-
      c(
        paste0(col_region, "_", col_date),
        col_name
      )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating incidence from ",
        use_column,
        " in column '",
        col_numerator,
        "' with population in column '",
        col_denominator,
        "' for ",
        N,
        " regions in column '",
        col_name,
        "' ... "
      ))
    }

    i <- 0

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <- input_data_i[
        order(input_data_i[[col_region]], input_data_i[[col_date]]),
      ]

      input_data_i_incidence <-
        input_data_i[[col_numerator]] /
        input_data_i[[col_denominator]] *
        pop_factor

      data_incidence_i <- data.frame(
        input_data_i[[paste0(col_region, "_", col_date)]],
        input_data_i_incidence
      )
      colnames(data_incidence_i) <- colnames(data_incidence)

      data_incidence <- rbind(
        data_incidence,
        data_incidence_i
      )
    }

    data_incidence <- data_incidence[
      !is.na(data_incidence[[paste0(col_region, "_", col_date)]]),
    ]

    input_data_incidence <-
      merge(
        input_data,
        data_incidence,
        by.x = paste0(col_region, "_", col_date),
        by.y = paste0(col_region, "_", col_date)
      )

    other_cols <- object@other_cols
    other_cols[["Incidence"]] <- col_name

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(input_data_incidence),
        data_statistics = object@data_statistics,
        index_col_names = object@index_col_names,
        cases_col_name = object@cases_col_name,
        other_cols = other_cols,
        timestamp = object@timestamp
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "calculate_incidence(infpan)",
      process = paste0(
        "Calculated incidence from ",
        use_column,
        " in column '",
        col_numerator,
        "' with population in column '",
        col_denominator,
        "' for ",
        N,
        " regions in column '",
        col_name,
        "' ... "
      )
    )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    invisible(infpan_object)
  }
)


setGeneric(
  "swash",
  function(
    object,
    verbose = FALSE
  ) {
    standardGeneric("swash")
  }
)

setMethod(
  "swash",
  "infpan",
  function(
    object,
    verbose = FALSE
  ) {
    infpan_object <- object

    sbm_object <- swash_backwash(
      infpan = infpan_object,
      verbose = verbose
    )

    invisible(sbm_object)
  }
)

setGeneric(
  "growth",
  function(
    object,
    S_iterations = 10,
    S_start_est_method = "bisect",
    seq_by = 10,
    nls = TRUE,
    add_constant = 1,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    standardGeneric("growth")
  }
)

setMethod(
  "growth",
  "infpan",
  function(
    object,
    S_iterations = 10,
    S_start_est_method = "bisect",
    seq_by = 10,
    nls = TRUE,
    add_constant = 1,
    overwrite = FALSE,
    verbose = FALSE
  ) {
    col_cases <- object@cases_col_name
    other_cols <- object@other_cols

    if ("Cum. cases" %in% names(other_cols)) {
      col_cum_cases <- other_cols[["Cum. cases"]]

      if (isFALSE(overwrite)) {
        message(
          paste0(
            "Cumulative infections already included in column '",
            col_cum_cases,
            "'"
          ),
          "\n"
        )
      } else {
        object <-
          calculate_cum(
            object,
            overwrite = overwrite,
            col_name = col_cum_cases,
            verbose = verbose
          )
      }
    } else {
      message("Cumulative infections not yet included", "\n")

      col_cum_cases <- paste0(col_cases, "_cum")

      object <-
        calculate_cum(
          object,
          col_name = col_cum_cases,
          verbose = verbose
        )
    }

    input_data <- object@input_data

    N <- object@data_statistics[1]

    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    time_format <- object@time_format

    N_names <- levels(as.factor(input_data[[col_region]]))

    logistic_growth_models <- list()

    results <- data.frame(matrix(ncol = 20))

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <-
        input_data_i[order(input_data_i[[col_date]]), ]

      input_data_cum <-
        data.frame(
          input_data_i[[col_cum_cases]],
          input_data_i[[col_date]]
        )
      colnames(input_data_cum) <- c("y", "t")

      min_date <- min(input_data_cum$t)
      max_date <- max(input_data_cum$t)

      logistic_growth_i <-
        logistic_growth(
          y = input_data_cum$y,
          t = input_data_cum$t,
          S = max(input_data_cum$y) * 1.01,
          S_start = NULL,
          S_end = NULL,
          S_iterations = S_iterations,
          S_start_est_method = S_start_est_method,
          seq_by = seq_by,
          nls = nls,
          add_constant = add_constant,
          verbose = verbose
        )

      results[i, 1] <- N_names[i]
      results[i, 2] <- min_date
      results[i, 3] <- max_date
      results[i, 4] <- logistic_growth_i@GrowthModel_OLS$S
      results[i, 5] <- logistic_growth_i@GrowthModel_OLS$r
      results[i, 6] <- logistic_growth_i@GrowthModel_OLS$y_0
      results[i, 7] <- logistic_growth_i@GrowthModel_OLS$ip
      results[i, 8] <- logistic_growth_i@GrowthModel_OLS$t_ip
      results[i, 9] <- logistic_growth_i@GrowthModel_OLS$fit_metrics$SQR
      results[i, 10] <- logistic_growth_i@GrowthModel_OLS$fit_metrics$R2
      results[i, 11] <- logistic_growth_i@GrowthModel_OLS$fit_metrics$MAPE

      if (isTRUE(logistic_growth_i@config$nls_estimation)) {
        results[i, 12] <- logistic_growth_i@GrowthModel_NLS$S
        results[i, 13] <- logistic_growth_i@GrowthModel_NLS$r
        results[i, 14] <- logistic_growth_i@GrowthModel_NLS$y_0
        results[i, 15] <- logistic_growth_i@GrowthModel_NLS$ip
        results[i, 16] <- logistic_growth_i@GrowthModel_NLS$t_ip
        results[i, 17] <- logistic_growth_i@GrowthModel_NLS$fit_metrics$SQR
        results[i, 18] <- logistic_growth_i@GrowthModel_NLS$fit_metrics$R2
        results[i, 19] <- logistic_growth_i@GrowthModel_NLS$fit_metrics$MAPE
      } else {
        results[i, 20] <- logistic_growth_i@config$nls_error_message
      }

      logistic_growth_models <- append(
        logistic_growth_models,
        logistic_growth_i
      )

      names(logistic_growth_models)[i] <- N_names[i]
    }

    colnames(results) <-
      c(
        "region",
        "min_date",
        "max_date",
        "S_OLS",
        "r_OLS",
        "y0_OLS",
        "ip_OLS",
        "tip_OLS",
        "SSQ_OLS",
        "R2_OLS",
        "MAPE_OLS",
        "S_NLS",
        "r_NLS",
        "y0_NLS",
        "ip_NLS",
        "tip_NLS",
        "SSQ_NLS",
        "R2_NLS",
        "MAPE_NLS",
        "nls_error_message"
      )

    growthmodels_object <-
      new(
        "growthmodels",
        results = results,
        growth_models = logistic_growth_models,
        model_type = "Logistic",
        results_cols = c(
          "region",
          "r_OLS",
          "tip_OLS",
          "R2_OLS",
          "r_NLS",
          "tip_NLS",
          "R2_NLS"
        ),
        results_cols_names = c(
          col_region,
          "OLS Growth rate",
          "OLS Inflection point",
          "OLS R-squared",
          "NLS Growth rate",
          "NLS Inflection point",
          "NLS R-squared"
        ),
        data_statistics = object@data_statistics,
        time_format = object@time_format
      )

    growthmodels_object <- add_timestamp(
      growthmodels_object,
      function_or_method = "growth",
      process = paste0(
        "Estimation of ",
        nrow(results),
        " logistic growth models and creation of growthmodels object"
      )
    )

    invisible(growthmodels_object)
  }
)

setGeneric(
  "growth_initial",
  function(
    object,
    time_units = 10,
    GI = 4,
    nls = TRUE,
    nls_start = list(a = 1, b = 0.1),
    add_constant = 1,
    verbose = FALSE
  ) {
    standardGeneric("growth_initial")
  }
)

setMethod(
  "growth_initial",
  "infpan",
  function(
    object,
    time_units = 10,
    GI = 4,
    nls = TRUE,
    nls_start = list(a = 1, b = 0.1),
    add_constant = 1,
    verbose = FALSE
  ) {
    N <- object@data_statistics[1]

    col_cases <- object@cases_col_name
    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    time_format <- object@time_format

    input_data <- object@input_data

    N_names <- levels(as.factor(input_data[[col_region]]))

    exponential_growth_models <- list()

    results <- data.frame(matrix(ncol = 16))

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      if (time_units > nrow(input_data_i)) {
        stop(paste0(
          "Dataset of region ",
          N_names[i],
          " contains ",
          nrow(input_data_i),
          " time units but ",
          time_units,
          " were stated."
        ))
      }

      input_data_i <-
        input_data_i[order(input_data_i[[col_date]]), ]

      input_data_cases <-
        data.frame(
          input_data_i[[col_cases]],
          input_data_i[[col_date]]
        )

      input_data_cases <- input_data_cases[1:time_units, ]
      colnames(input_data_cases) <- c("y", "t")

      if (nrow(input_data_cases) > 1) {
        min_date <- min(input_data_cases$t)
        max_date <- max(input_data_cases$t)

        exponential_growth_i <-
          exponential_growth(
            y = input_data_cases$y,
            t = input_data_cases$t,
            GI = GI,
            nls = nls,
            nls_start = nls_start,
            add_constant = add_constant,
            verbose = verbose
          )

        results[i, 1] <- N_names[i]
        results[i, 2] <- as.Date(min_date, format = time_format)
        results[i, 3] <- as.Date(max_date, format = time_format)
        results[i, 4] <- exponential_growth_i@GrowthModel_OLS$exp_gr
        results[i, 5] <- exponential_growth_i@GrowthModel_OLS$R0
        results[i, 6] <- exponential_growth_i@GrowthModel_OLS$doubling
        results[i, 7] <- exponential_growth_i@GrowthModel_OLS$fit_metrics$SQR
        results[i, 8] <- exponential_growth_i@GrowthModel_OLS$fit_metrics$R2
        results[i, 9] <- exponential_growth_i@GrowthModel_OLS$fit_metrics$MAPE

        if (isTRUE(exponential_growth_i@config$nls_estimation)) {
          results[i, 10] <- exponential_growth_i@GrowthModel_NLS$exp_gr
          results[i, 11] <- exponential_growth_i@GrowthModel_NLS$R0
          results[i, 12] <- exponential_growth_i@GrowthModel_NLS$doubling
          results[i, 13] <- exponential_growth_i@GrowthModel_NLS$fit_metrics$SQR
          results[i, 14] <- exponential_growth_i@GrowthModel_NLS$fit_metrics$R2
          results[
            i,
            15
          ] <- exponential_growth_i@GrowthModel_NLS$fit_metrics$MAPE
        } else {
          results[i, 16] <- exponential_growth_i@config$nls_error_message
        }

        exponential_growth_models <-
          append(
            exponential_growth_models,
            exponential_growth_i
          )

        exponential_growth_models[[N_names[i]]] <- exponential_growth_i
      } else {
        warning(paste0(
          "No cases for region ",
          N_names[i],
          ". No exponential growth model built."
        ))

        results[i, 1] <- N_names[i]
        results[i, 2:15] <- NA
      }
    }

    colnames(results) <-
      c(
        "region",
        "min_date",
        "max_date",
        "r_OLS",
        "R_0_OLS",
        "doubling_rate_OLS",
        "SSQ_OLS",
        "R2_OLS",
        "MAPE_OLS",
        "r_NLS",
        "R_0_NLS",
        "doubling_rate_NLS",
        "SSQ_NLS",
        "R2_NLS",
        "MAPE_NLS"
      )

    growthmodels_object <-
      new(
        "growthmodels",
        results = results,
        growth_models = exponential_growth_models,
        model_type = "Exponential",
        results_cols = c(
          "region",
          "r_OLS",
          "R_0_OLS",
          "doubling_rate_OLS",
          "R2_OLS",
          "r_NLS",
          "R_0_NLS",
          "doubling_rate_NLS",
          "R2_NLS"
        ),
        results_cols_names = c(
          col_region,
          "OLS Growth rate",
          "OLS Basic reproduction number",
          "OLS Doubling rate",
          "OLS R-squared",
          "NLS Growth rate",
          "NLS Basic reproduction number",
          "NLS Doubling rate",
          "NLS R-squared"
        ),
        data_statistics = object@data_statistics,
        time_format = object@time_format
      )

    growthmodels_object <- add_timestamp(
      growthmodels_object,
      function_or_method = "growth_initial",
      process = "Analysis of exponential growth and creation of growthmodels object"
    )

    invisible(growthmodels_object)
  }
)


setGeneric(
  "growth_breaks",
  function(
    object,
    ln = FALSE,
    add_constant = 1,
    alpha = 0.05,
    verbose = FALSE
  ) {
    standardGeneric("growth_breaks")
  }
)

setMethod(
  "growth_breaks",
  "infpan",
  function(
    object,
    ln = FALSE,
    add_constant = 1,
    alpha = 0.05,
    verbose = FALSE
  ) {
    N <- object@data_statistics[1]

    col_cases <- object@cases_col_name
    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    time_format <- object@time_format

    input_data <- object@input_data

    N_names <- levels(as.factor(input_data[[col_region]]))

    breaks_growth_models <- list()

    results <- data.frame(matrix(ncol = 21))
    colnames(results) <-
      c(
        "region",
        "min_date",
        "max_date",
        "first",
        "last",
        "intercept",
        "intercept_CI25",
        "intercept_CI975",
        "intercept_p",
        "slope",
        "slope_CI25",
        "slope_CI975",
        "slope_p",
        "SQR",
        "SAR",
        "SQT",
        "R2",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE"
      )

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <-
        input_data_i[order(input_data_i[[col_date]]), ]

      input_data_cases <-
        data.frame(
          input_data_i[[col_cases]],
          input_data_i[[col_date]]
        )

      colnames(input_data_cases) <- c("y", "t")

      if (nrow(input_data_cases) > 1) {
        min_date <- min(input_data_cases$t)
        max_date <- max(input_data_cases$t)

        breaks_growth_i <-
          breaks_growth(
            y = input_data_cases$y,
            t = input_data_cases$t,
            ln = ln,
            add_constant = add_constant,
            alpha = alpha,
            verbose = verbose
          )

        results_i <- data.frame(
          breaks_growth_i@GrowthModel_OLS$segments_models_df
        )

        results_i$region <- N_names[i]
        results_i$min_date <- as.Date(min_date, format = time_format)
        results_i$max_date <- as.Date(max_date, format = time_format)

        results_i <- results_i[c(19:21, 1:18)]

        results <-
          rbind(
            results,
            results_i
          )

        breaks_growth_models <-
          append(
            breaks_growth_models,
            breaks_growth_i
          )

        breaks_growth_models[[N_names[i]]] <- breaks_growth_i
      } else {
        warning(paste0(
          "No cases for region ",
          N_names[i],
          ". No breaking points model built."
        ))

        results[i, 1] <- N_names[i]
        results[i, 2:21] <- NA
      }
    }

    colnames(results) <-
      c(
        "region",
        "min_date",
        "max_date",
        "first",
        "last",
        "intercept",
        "intercept_CI25",
        "intercept_CI975",
        "intercept_p",
        "slope",
        "slope_CI25",
        "slope_CI975",
        "slope_p",
        "SQR",
        "SAR",
        "SQT",
        "R2",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE"
      )

    results <- results[!is.na(results$region), ]

    growthmodels_object <-
      new(
        "growthmodels",
        results = results,
        growth_models = breaks_growth_models,
        model_type = "Breaking Points",
        results_cols = c(
          "region",
          "first",
          "last",
          "intercept",
          "intercept_p",
          "slope",
          "slope_p",
          "R2"
        ),
        results_cols_names = c(
          col_region,
          "First of segment",
          "Last of segment",
          "Intercept",
          "Intercept p value",
          "Slope",
          "Slope p value",
          "R-squared"
        ),
        data_statistics = object@data_statistics,
        time_format = object@time_format
      )

    growthmodels_object <- add_timestamp(
      growthmodels_object,
      function_or_method = "growth_breaks",
      process = "Analysis of breaking points and creation of growthmodels object"
    )

    invisible(growthmodels_object)
  }
)


setGeneric(
  "growth_hawkes",
  function(
    object,
    optim_method = "L-BFGS-B",
    verbose = FALSE
  ) {
    standardGeneric("growth_hawkes")
  }
)

setMethod(
  "growth_hawkes",
  "infpan",
  function(
    object,
    optim_method = "L-BFGS-B",
    verbose = FALSE
  ) {
    N <- object@data_statistics[1]

    col_cases <- object@cases_col_name
    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    time_format <- object@time_format

    input_data <- object@input_data

    N_names <- levels(as.factor(input_data[[col_region]]))

    hawkes_growth_models <- list()

    results <- data.frame(matrix(ncol = 7))

    for (i in 1:N) {
      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]

      input_data_i <-
        input_data_i[order(input_data_i[[col_date]]), ]

      input_data_cases <-
        data.frame(
          input_data_i[[col_cases]],
          input_data_i[[col_date]]
        )

      colnames(input_data_cases) <- c("y", "t")

      if (nrow(input_data_cases) > 1) {
        min_date <- min(input_data_cases$t)
        max_date <- max(input_data_cases$t)

        hawkes_growth_i <-
          hawkes_growth(
            y = input_data_cases$y,
            verbose = verbose
          )

        results[i, 1] <- N_names[i]
        results[i, 2] <- as.Date(min_date, format = time_format)
        results[i, 3] <- as.Date(max_date, format = time_format)
        results[i, 4] <- hawkes_growth_i@mu
        results[i, 5] <- hawkes_growth_i@alpha
        results[i, 6] <- hawkes_growth_i@beta
        results[i, 7] <- hawkes_growth_i@br

        hawkes_growth_models <-
          append(
            hawkes_growth_models,
            hawkes_growth_i
          )

        hawkes_growth_models[[N_names[i]]] <- hawkes_growth_i
      } else {
        warning(paste0(
          "No cases for region ",
          N_names[i],
          ". No Hawkes growth model built."
        ))

        results[i, 1] <- N_names[i]
        results[i, 2] <- NA
        results[i, 3] <- NA
        results[i, 4] <- NA
        results[i, 5] <- NA
        results[i, 6] <- NA
        results[i, 7] <- NA
      }
    }

    colnames(results) <-
      c(
        "region",
        "min_date",
        "max_date",
        "mu",
        "alpha",
        "beta",
        "br"
      )

    growthmodels_object <-
      new(
        "growthmodels",
        results = results,
        growth_models = hawkes_growth_models,
        model_type = model_descriptions[["hawkes"]],
        results_cols = c(
          "region",
          "mu",
          "alpha",
          "beta",
          "br"
        ),
        results_cols_names = c(
          col_region,
          "Baseline",
          "Excitation",
          "Decay",
          "Breaking ratio"
        ),
        data_statistics = object@data_statistics,
        time_format = object@time_format
      )

    growthmodels_object <- add_timestamp(
      growthmodels_object,
      function_or_method = "growth_hawkes",
      process = "Analysis of Hawkes growth and creation of growthmodels object"
    )

    invisible(growthmodels_object)
  }
)

setMethod(
  "plot",
  signature(x = "infpan"),
  function(
    x,
    y = NULL,
    col = "red",
    lty = "solid",
    scale = FALSE,
    normalize_by_col = NULL,
    normalize_factor = 1,
    plot_rollmean = FALSE,
    rollmean_col = "blue",
    rollmean_lty = "solid",
    rollmean_k = 7,
    rollmean_align = "center",
    rollmean_fill = NA,
    growth_col = "orange",
    growth_lty = "solid",
    growth_per_time_unit = 1
  ) {
    object <- x

    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    N <- object@data_statistics[1]

    plot_cols <- 4
    plot_rows <- ceiling(N / plot_cols)
    par(mfrow = c(plot_rows, plot_cols))

    input_data <- object@input_data

    col_cases <- object@cases_col_name
    col_region <- object@index_col_names[1]
    col_date <- object@index_col_names[2]

    N_names <- levels(as.factor(input_data[[col_region]]))

    if (!is.null(normalize_by_col)) {
      input_data[[paste0(col_cases, "_normalized")]] <-
        input_data[[col_cases]] /
        input_data[[normalize_by_col]] *
        normalize_factor

      y_max <- max(input_data[[paste0(col_cases, "_normalized")]]) * 1.05
    } else {
      y_max <- max(input_data[[col_cases]]) * 1.05
    }

    i <- 0

    for (i in 1:N) {
      par(mar = c(2, 2, 1, 1))

      input_data_i <-
        input_data[input_data[[col_region]] == N_names[i], ]
      input_data_i <-
        input_data_i[order(input_data_i[[col_date]]), ]

      if (!is.null(normalize_by_col)) {
        if (!scale) {
          y_max <- max(input_data_i[[paste0(col_cases, "_normalized")]]) * 1.05
        }

        plot(
          input_data_i[[col_date]],
          input_data_i[[paste0(col_cases, "_normalized")]],
          type = "l",
          col = col,
          lty = lty,
          main = N_names[i],
          ylim = c(0, y_max)
        )

        if (isTRUE(plot_rollmean)) {
          input_data_i[[paste0(col_cases, "_normalized_rm")]] <-
            rollmean(
              input_data_i[[paste0(col_cases, "_normalized")]],
              k = rollmean_k,
              fill = rollmean_fill,
              align = rollmean_align
            )

          lines(
            input_data_i[[col_date]],
            input_data_i[[paste0(col_cases, "_normalized_rm")]],
            col = rollmean_col,
            lty = rollmean_lty
          )
        }
      } else {
        if (!scale) {
          y_max <- max(input_data_i[[col_cases]]) * 1.05
        }

        plot(
          input_data_i[[col_date]],
          input_data_i[[col_cases]],
          col = col,
          main = N_names[i],
          type = "l",
          ylim = c(0, y_max)
        )

        if (isTRUE(plot_rollmean)) {
          input_data_i[[paste0(col_cases, "_rm")]] <-
            rollmean(
              input_data_i[[col_cases]],
              k = rollmean_k,
              fill = rollmean_fill,
              align = rollmean_align
            )

          lines(
            input_data_i[[col_date]],
            input_data_i[[paste0(col_cases, "_rm")]],
            col = rollmean_col,
            lty = rollmean_lty
          )
        }
      }
    }

    invisible(object)
  }
)


# Class sbm:
setClass(
  "sbm",
  slots = list(
    R_0A = "numeric",
    integrals = "numeric",
    velocity = "numeric",
    occ_regions = "data.frame",
    SIR_regions = "data.frame",
    cases_by_date = "data.frame",
    cases_by_region = "data.frame",
    input_data = "data.frame",
    data_statistics = "numeric",
    col_names = "character",
    timestamp = "list"
  ),
  prototype = list(
    timestamp = list()
  )
)

# Methods for class sbm:
setMethod(
  "summary",
  "sbm",
  function(object) {
    cat(model_descriptions[["sbm"]], "\n\n")

    cat("Integrals\n")
    cat(sprintf("  Susceptible areas : %.3f\n", object@integrals[1]))
    cat(sprintf("  Infected areas    : %.3f\n", object@integrals[2]))
    cat(sprintf("  Recovered areas   : %.3f\n\n", object@integrals[3]))

    cat("Velocity\n")
    cat(sprintf("  Leading edge      : %.3f\n", object@velocity[1]))
    cat(sprintf("  Following edge    : %.3f\n\n", object@velocity[2]))

    cat(sprintf("Spatial reproduction number : %.3f\n\n", object@R_0A))

    cat("Input data\n")
    cat(sprintf("  Units       : %s\n", object@data_statistics[1]))
    cat(sprintf("  No-case     : %s\n", object@data_statistics[3]))
    cat(sprintf("  Time points : %s\n", object@data_statistics[2]))
    cat(sprintf(
      "  Balanced    : %s\n",
      ifelse(object@data_statistics[4], "YES", "NO")
    ))
  }
)

setMethod(
  "print",
  "sbm",
  function(x) {
    cat(
      paste0(
        model_descriptions[["sbm"]],
        " with ",
        x@data_statistics[1],
        " spatial units and ",
        x@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")

    invisible(x)
  }
)

setMethod(
  "show",
  "sbm",
  function(object) {
    cat(
      paste0(
        model_descriptions[["sbm"]],
        " with ",
        object@data_statistics[1],
        " spatial units and ",
        object@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")

    invisible((object))
  }
)

setMethod(
  "plot",
  "sbm",
  function(
    x,
    y = NULL,
    col_edges = "blue",
    xlab_edges = "Time",
    ylab_edges = "Regions",
    main_edges = "Edges",
    col_SIR = c("blue", "red", "green"),
    lty_SIR = c("solid", "solid", "solid"),
    lwd_SIR = c(1, 1, 1),
    xlab_SIR = "Time",
    ylab_SIR = "Regions",
    main_SIR = "SIR integrals",
    col_cases = "red",
    lty_cases = "solid",
    lwd_cases = 1,
    xlab_cases = "Time",
    ylab_cases = "Infections",
    main_cases = "Daily infections",
    xlab_cum = "Cases",
    ylab_cum = "Regions",
    main_cum = "Cumulative infections per region",
    horiz_cum = TRUE,
    separate_plots = FALSE
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    if (!separate_plots) {
      par(mfrow = c(2, 2))
    }

    barplot(
      x@occ_regions$LE_FE,
      col = col_edges,
      xlab = xlab_edges,
      ylab = ylab_edges,
      main = main_edges
    )

    plot(
      x = as.Date(x@SIR_regions$date),
      y = x@SIR_regions$susceptible,
      col = col_SIR[1],
      xlab = xlab_SIR,
      ylab = ylab_SIR,
      "l",
      lty = lty_SIR[1],
      lwd = lwd_SIR[1],
      ylim = c(0, x@data_statistics[1] + 1),
      main = main_SIR
    )
    lines(
      x = as.Date(x@SIR_regions$date),
      y = x@SIR_regions$infected,
      col = col_SIR[2],
      lty = lty_SIR,
      lwd = lwd_SIR[2]
    )
    lines(
      x = as.Date(x@SIR_regions$date),
      y = x@SIR_regions$recovered,
      col = col_SIR[3],
      lty = lty_SIR[3],
      lwd = lwd_SIR[3]
    )

    plot(
      x@cases_by_date$date,
      x@cases_by_date$cases,
      type = "l",
      lty = lty_cases,
      lwd = lwd_cases,
      col = col_cases,
      xlab = xlab_cases,
      ylab = ylab_cases,
      main = main_cases
    )

    x@cases_by_region <- x@cases_by_region[
      order(x@cases_by_region$cases_cumulative),
    ]
    barplot(
      height = x@cases_by_region$cases_cumulative,
      horiz = horiz_cum,
      names.arg = x@cases_by_region$region,
      xlab = xlab_cum,
      ylab = ylab_cum,
      main = main_cum,
      las = 1
    )

    invisible(x)
  }
)

setMethod(
  "confint",
  "sbm",
  function(
    object,
    iterations = 100,
    samples_ratio = 0.8,
    alpha = 0.05,
    replace = TRUE,
    verbose = FALSE
  ) {
    N <- object@data_statistics[1]
    TP <- object@data_statistics[2]
    input_data <- object@input_data
    obs <- nrow(object@input_data)

    regions <- as.character(levels(as.factor(object@input_data[[object@col_names[
      3
    ]]])))
    regions_to_sample <- round(N * samples_ratio)

    bootstrap_config <- list(
      iterations = iterations,
      samples_ratio = samples_ratio,
      regions_to_sample = regions_to_sample,
      alpha = alpha,
      replace = replace
    )

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating bootstrap confidence intervals with ",
        iterations,
        " iterations and alpha = ",
        alpha
      ))
    }

    i <- 0
    swash_bootstrap <- matrix(ncol = 9, nrow = iterations)

    for (i in 1:iterations) {
      regions_sample <-
        sample(
          x = regions,
          size = regions_to_sample,
          replace = replace
        )

      data_sample <-
        input_data[
          as.character(input_data[[object@col_names[3]]]) %in% regions_sample,
        ]

      data_sample_size <- nrow(data_sample)

      data_sample_swash <-
        swash_backwash(
          data = data_sample,
          col_cases = object@col_names[1],
          col_date = object@col_names[2],
          col_region = object@col_names[3]
        )

      swash_bootstrap[i, 1] <- i
      swash_bootstrap[i, 2] <- data_sample_swash@integrals[1]
      swash_bootstrap[i, 3] <- data_sample_swash@integrals[2]
      swash_bootstrap[i, 4] <- data_sample_swash@integrals[3]
      swash_bootstrap[i, 5] <- data_sample_swash@velocity[1]
      swash_bootstrap[i, 6] <- data_sample_swash@velocity[2]
      swash_bootstrap[i, 7] <- data_sample_swash@velocity[3]
      swash_bootstrap[i, 8] <- data_sample_swash@R_0A
      swash_bootstrap[i, 9] <- data_sample_size
    }

    colnames(swash_bootstrap) <-
      c(
        "iteration",
        "S_A",
        "I_A",
        "R_A",
        "t_LE",
        "t_FE",
        "t_FE-t_LE",
        "R_0A",
        "sample_size"
      )

    ci_lower <- alpha / 2
    ci_upper <- 1 - (alpha / 2)

    S_A_ci <- quantile(swash_bootstrap[, 2], probs = c(ci_lower, ci_upper))
    I_A_ci <- quantile(swash_bootstrap[, 3], probs = c(ci_lower, ci_upper))
    R_A_ci <- quantile(swash_bootstrap[, 4], probs = c(ci_lower, ci_upper))
    integrals_ci <- list(S_A_ci = S_A_ci, I_A_ci = I_A_ci, R_A_ci = R_A_ci)

    t_LE_ci <- quantile(swash_bootstrap[, 5], probs = c(ci_lower, ci_upper))
    t_FE_ci <- quantile(swash_bootstrap[, 6], probs = c(ci_lower, ci_upper))
    t_FE_t_LE_ci <- quantile(
      swash_bootstrap[, 7],
      probs = c(ci_lower, ci_upper)
    )
    velocity_ci <- list(
      t_LE_ci = t_LE_ci,
      t_FE_ci = t_FE_ci,
      t_FE_t_LE_ci = t_FE_t_LE_ci
    )

    R_0A_ci <- quantile(swash_bootstrap[, 8], probs = c(ci_lower, ci_upper))

    if (isTRUE(verbose)) {
      print("OK", "\n")
    }

    new(
      "sbm_ci",
      R_0A = object@R_0A,
      integrals = object@integrals,
      velocity = object@velocity,
      occ_regions = object@occ_regions,
      cases_by_date = object@cases_by_date,
      input_data = object@input_data,
      data_statistics = object@data_statistics,
      col_names = object@col_names,
      integrals_ci = integrals_ci,
      velocity_ci = velocity_ci,
      R_0A_ci = R_0A_ci,
      iterations = data.frame(swash_bootstrap),
      ci = c(ci_lower, ci_upper),
      config = bootstrap_config
    )
  }
)


# Class sbm_ci:
setClass(
  "sbm_ci",
  slots = list(
    R_0A = "numeric",
    integrals = "numeric",
    velocity = "numeric",
    occ_regions = "data.frame",
    cases_by_date = "data.frame",
    cases_by_region = "data.frame",
    input_data = "data.frame",
    data_statistics = "numeric",
    col_names = "character",
    integrals_ci = "list",
    velocity_ci = "list",
    R_0A_ci = "numeric",
    iterations = "data.frame",
    ci = "numeric",
    config = "list"
  )
)

# Methods of class sbm_ci:
setMethod(
  "summary",
  "sbm_ci",
  function(object) {
    cat("Confidence Intervals for Swash-Backwash Model\n\n")

    cat("Integrals\n")
    cat(
      sprintf(
        "  Susceptible areas : [%.3f, %.3f]\n",
        object@integrals_ci$S_A_ci[1],
        object@integrals_ci$S_A_ci[2]
      )
    )
    cat(
      sprintf(
        "  Infected areas    : [%.3f, %.3f]\n",
        object@integrals_ci$I_A_ci[1],
        object@integrals_ci$I_A_ci[2]
      )
    )
    cat(
      sprintf(
        "  Recovered areas   : [%.3f, %.3f]\n\n",
        object@integrals_ci$R_A_ci[1],
        object@integrals_ci$R_A_ci[2]
      )
    )

    cat("Velocity\n")
    cat(
      sprintf(
        "  Leading edge      : [%.3f, %.3f]\n",
        object@velocity_ci$t_LE_ci[1],
        object@velocity_ci$t_LE_ci[2]
      )
    )
    cat(
      sprintf(
        "  Following edge    : [%.3f, %.3f]\n\n",
        object@velocity_ci$t_FE_ci[1],
        object@velocity_ci$t_FE_ci[2]
      )
    )

    cat(
      sprintf(
        "Spatial reproduction number : [%.3f, %.3f]\n\n",
        object@R_0A_ci[1],
        object@R_0A_ci[2]
      )
    )

    cat("Configuration for confidence intervals\n")
    cat(
      sprintf(
        "  CI alpha    : %s\n",
        object@config$alpha
      )
    )
    cat(sprintf(
      "  Sample      : %s %% (%s units)\n",
      object@config$samples_ratio * 100,
      object@config$regions_to_sample
    ))
    cat(sprintf("  Iterations  : %s\n", object@config$iterations))
    cat(sprintf(
      "  Bootstrap   : %s\n",
      ifelse(object@config$replace, "YES", "NO")
    ))

    cat("\nInput data\n")
    cat(sprintf("  Units       : %s\n", object@data_statistics[1]))
    cat(sprintf("  No-case     : %s\n", object@data_statistics[3]))
    cat(sprintf("  Time points : %s\n", object@data_statistics[2]))
    cat(sprintf(
      "  Balanced    : %s\n",
      ifelse(object@data_statistics[4], "YES", "NO")
    ))
  }
)

setMethod(
  "print",
  "sbm_ci",
  function(x) {
    cat(
      paste0(
        "Confidence Intervals for Swash-Backwash Model with ",
        x@data_statistics[1],
        " spatial units and ",
        x@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat(
      paste0(
        "Resampling of ",
        x@config$regions_to_sample,
        " spatial units (",
        x@config$samples_ratio * 100,
        " %) with ",
        x@config$iterations,
        " iterations"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")
  }
)


setMethod(
  "show",
  "sbm_ci",
  function(object) {
    cat(
      paste0(
        "Confidence Intervals for Swash-Backwash Model with ",
        object@data_statistics[1],
        " spatial units and ",
        object@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat(
      paste0(
        "Resampling of ",
        object@config$regions_to_sample,
        " spatial units (",
        object@config$samples_ratio * 100,
        " %) with ",
        object@config$iterations,
        " iterations"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")
  }
)


setMethod(
  "plot",
  "sbm_ci",
  function(
    x,
    y = NULL,
    col_bars = "grey",
    col_ci = "red"
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    par(mfrow = c(2, 3))

    alpha <- x@config$alpha

    hist_ci(
      x@iterations$S_A,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Susceptible areas integral",
      xlab = "S_A",
      ylab = "Frequency"
    )
    hist_ci(
      x@iterations$I_A,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Infected areas integral",
      xlab = "I_A",
      ylab = "Frequency"
    )
    hist_ci(
      x@iterations$R_A,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Recovered areas integral",
      xlab = "R_A",
      ylab = "Frequency"
    )

    hist_ci(
      x@iterations$t_LE,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Leading edge",
      xlab = "t_LE",
      ylab = "Frequency"
    )
    hist_ci(
      x@iterations$t_FE,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Following edge",
      xlab = "t_FE",
      ylab = "Frequency"
    )

    hist_ci(
      x@iterations$R_0A,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = "Spatial reproduction number",
      xlab = "R_0A",
      ylab = "Frequency"
    )

    invisible(x)
  }
)


# Class countries:
setClass(
  "countries",
  slots = list(
    sbm_ci1 = "sbm_ci",
    sbm_ci2 = "sbm_ci",
    D = "numeric",
    D_ci = "numeric",
    config = "list",
    country_names = "character",
    indicator = "character"
  )
)

# Methods for class countries:
setMethod(
  "plot",
  "countries",
  function(
    x,
    y = NULL,
    col_bars = "grey",
    col_ci = "red"
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    par(mfrow = c(2, 2))

    alpha <- x@config$alpha

    indicator <- x@indicator

    hist_ci(
      x@sbm_ci1@iterations[[indicator]],
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = paste0("Indicator ", indicator, " for ", x@country_names[1]),
      xlab = indicator,
      ylab = "Frequency"
    )

    hist_ci(
      x@sbm_ci2@iterations[[indicator]],
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = paste0("Indicator ", indicator, " for ", x@country_names[2]),
      xlab = indicator,
      ylab = "Frequency"
    )

    country1 <- data.frame(
      x@sbm_ci1@iterations[[indicator]],
      x@country_names[1]
    )
    colnames(country1) <-
      c(indicator, "country")
    country2 <- data.frame(
      x@sbm_ci2@iterations[[indicator]],
      x@country_names[2]
    )
    colnames(country2) <-
      c(indicator, "country")

    iterations_countries <-
      rbind(
        country1,
        country2
      )

    boxplot(
      iterations_countries[[indicator]] ~ iterations_countries$country,
      xlab = "Country",
      ylab = indicator
    )

    hist_ci(
      x@D,
      alpha = alpha,
      col_bars = col_bars,
      col_ci = col_ci,
      main = paste0("Difference in indicator ", indicator),
      xlab = paste0("D (", indicator, " )"),
      ylab = "Frequency"
    )

    invisible(x)
  }
)

setMethod(
  "summary",
  "countries",
  function(object) {
    D_ci <- object@D_ci
    indicator <- object@indicator

    D_mean <- mean(object@D, na.rm = TRUE)
    D_median <- median(object@D, na.rm = TRUE)

    cat("Two-country comparison for Swash-Backwash Model\n\n")

    cat(
      sprintf(
        "Difference in %s\n",
        indicator
      )
    )
    cat(
      sprintf(
        "  Mean   : %.3f\n",
        D_mean
      )
    )
    cat(
      sprintf(
        "  Median : %.3f\n",
        D_median
      )
    )
    cat(
      sprintf(
        "  Confidence interval : [%.3f, %.3f]\n\n",
        D_ci[1],
        D_ci[2]
      )
    )

    cat("Configuration for confidence intervals\n")
    cat(
      sprintf(
        "  CI alpha    : %s\n",
        object@config$alpha
      )
    )
    cat(
      sprintf(
        "  Sample      : %s %%\n",
        object@config$samples_ratio * 100
      )
    )
    cat(
      sprintf(
        "  Iterations  : %s\n",
        object@config$iterations
      )
    )
    cat(
      sprintf(
        "  Bootstrap   : %s\n",
        ifelse(object@config$replace, "YES", "NO")
      )
    )
  }
)

setMethod(
  "show",
  "countries",
  function(object) {
    cat(paste0("Two-country comparison with Swash-Backwash Model"), "\n")
    cat("Use summary() for results", "\n")
  }
)

setMethod(
  "print",
  "countries",
  function(x) {
    cat(paste0("Two-country comparison with Swash-Backwash Model"), "\n")
    cat("Use summary() for results", "\n")
  }
)


R_t <-
  function(
    infections,
    GP = 4,
    correction = FALSE
  ) {
    i <- 0

    infections_daily_A <- vector()
    infections_daily_B <- vector()

    R_t <- vector()
    R_t[1:(GP - 1)] <- NA

    for (i in (GP * 2):length(infections)) {
      infections_daily_A[i] <- sum(infections[(i - (GP - 1)):i])
      infections_daily_B[i] <- sum(infections[(i - ((GP * 2) - 1)):(i - GP)])

      if (correction) {
        if (infections_daily_B[i] < 1) {
          infections_daily_B[i] <- 1
        }
      }

      R_t[i] <- as.numeric(infections_daily_A[i] / infections_daily_B[i])
    }

    results <- list(
      R_t = R_t,
      infections_data = cbind.data.frame(
        infections_daily_A,
        infections_daily_B,
        R_t
      )
    )

    return(results)
  }


quantile_ci <-
  function(
    x,
    alpha = 0.05
  ) {
    ci_lower <- alpha / 2
    ci_upper <- 1 - (alpha / 2)

    ci <-
      quantile(
        x,
        probs = c(ci_lower, ci_upper)
      )

    return(ci)
  }


hist_ci <-
  function(
    x,
    alpha = 0.05,
    col_bars = "grey",
    col_ci = "red",
    ...
  ) {
    ci <- quantile_ci(
      x = x,
      alpha = alpha
    )

    hist(x, col = col_bars, ...)
    abline(v = ci[1], col = col_ci)
    abline(v = ci[2], col = col_ci)
    abline(v = median(x), col = col_ci)
  }


is_balanced <-
  function(
    data,
    col_cases,
    col_date,
    col_region,
    as_balanced = TRUE,
    fill_missing = 0
  ) {
    N <- nlevels(as.factor(data[[col_region]]))
    TP <- nlevels(as.factor(data[[col_date]]))

    if (nrow(data) != (TP * N)) {
      data_balanced <- FALSE
    } else {
      if (
        (length(unique(table(data[[col_date]]))) != 1) |
          length(unique(table(data[[col_region]]))) != 1
      ) {
        data_balanced <- FALSE
      } else {
        if (anyNA(data[[col_cases]])) {
          data_balanced <- FALSE
        } else {
          data_balanced <- TRUE
        }
      }
    }

    if ((!data_balanced) & (as_balanced)) {
      data <-
        as_balanced(
          data,
          col_cases,
          col_date,
          col_region,
          fill_missing = fill_missing
        )

      data_balanced <- TRUE
    }

    results <-
      list(
        data_balanced = data_balanced,
        data = data
      )

    return(results)
  }


as_balanced <-
  function(
    data,
    col_cases,
    col_date,
    col_region,
    fill_missing = 0
  ) {
    N <- nlevels(as.factor(data[[col_region]]))
    TP <- nlevels(as.factor(data[[col_date]]))
    N_names <- as.character(levels(as.factor(data[[col_region]])))
    TP_t <- as.character(levels(as.factor(data[[col_date]])))

    N_x_TPt <- merge(N_names, TP_t)
    colnames(N_x_TPt) <- c(
      paste0("__", col_region, "__"),
      paste0("__", col_date, "__")
    )
    N_x_TPt[[paste0("__", col_region, "_x_", col_date, "__")]] <-
      paste0(
        N_x_TPt[[paste0("__", col_region, "__")]],
        "_x_",
        N_x_TPt[[paste0("__", col_date, "__")]]
      )

    data[[paste0("__", col_region, "_x_", col_date, "__")]] <-
      paste0(data[[col_region]], "_x_", data[[col_date]])

    data <-
      merge(
        data,
        N_x_TPt,
        by.x = paste0("__", col_region, "_x_", col_date, "__"),
        by.y = paste0("__", col_region, "_x_", col_date, "__")
      )

    data[[col_region]] <- data[[paste0("__", col_region, "__")]]
    data[[col_date]] <- data[[paste0("__", col_date(), "__")]]
    if (nrow(data[is.na(data[[col_cases]]), ]) > 0) {
      data[is.na(data[[col_cases]]), ][[col_cases]] <- fill_missing
    }

    data[[paste0("__", col_region, "_x_", col_date, "__")]] <- NULL
    data[[paste0("__", col_region, "__")]] <- NULL
    data[[paste0("__", col_date, "__")]] <- NULL

    return(data)
  }


setClass(
  "expgrowth",
  slots = list(
    GrowthModel_OLS = "list",
    GrowthModel_NLS = "list",
    y = "numeric",
    t = "numeric",
    config = "list"
  )
)

setMethod(
  "summary",
  "expgrowth",
  function(object) {
    GrowthModel_OLS <- object@GrowthModel_OLS
    GrowthModel_NLS <- object@GrowthModel_NLS
    config <- object@config

    cat(model_descriptions[["expgrowth"]], "\n\n")

    cat("OLS model\n")
    cat(sprintf("  Growth rate              : %.3f\n", GrowthModel_OLS$exp_gr))
    cat(sprintf("  Baseline                 : %.3f\n", GrowthModel_OLS$y_0))
    cat(sprintf("  Basic reproduction number: %.3f\n", GrowthModel_OLS$R0))
    cat(sprintf(
      "  Doubling rate            : %.3f\n",
      GrowthModel_OLS$doubling
    ))
    cat(sprintf(
      "  R-squared                : %.3f\n\n",
      GrowthModel_OLS$fit_metrics$R2
    ))

    cat("NLS model\n")
    cat(sprintf("  Growth rate              : %.3f\n", GrowthModel_NLS$exp_gr))
    cat(sprintf("  Baseline                 : %.3f\n", GrowthModel_NLS$y_0))
    cat(sprintf("  Basic reproduction number: %.3f\n", GrowthModel_NLS$R0))
    cat(sprintf(
      "  Doubling rate            : %.3f\n",
      GrowthModel_NLS$doubling
    ))
    cat(sprintf(
      "  R-squared                : %.3f\n\n",
      GrowthModel_NLS$fit_metrics$R2
    ))

    cat("Input\n")

    cat(sprintf(
      "  Time points (not NaN)    : %.0f\n",
      as.integer(object@config$TP)
    ))

    invisible(object)
  }
)

setMethod(
  "plot",
  "expgrowth",
  function(
    x,
    y = NULL,
    cp_col = "lightblue",
    cp_pch = 19,
    cl_col = "red",
    x_lab = "Time",
    y_lab = "Daily infections",
    plot_title = "Exponential growth model",
    text_size = 1,
    anno_size = 0.7,
    bgrid = TRUE,
    bgrid_col = "white",
    bgrid_type = 1,
    bgrid_size = 1,
    bg_col = "white",
    show_model_results = TRUE
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    t <- x@t
    y <- x@y
    GrowthModel_NLS <- x@GrowthModel_NLS
    GrowthModel_OLS <- x@GrowthModel_OLS

    if (is.null(GrowthModel_NLS)) {
      y_pred <- GrowthModel_OLS$y_pred
      exp_gr <- GrowthModel_OLS$exp_gr
      y_0 <- GrowthModel_OLS$y_0
      R0 <- GrowthModel_OLS$R0
      doubling <- GrowthModel_OLS$doubling
      sum_of_squares <- GrowthModel_OLS$fit_metrics$SQT
      R2 <- GrowthModel_OLS$fit_metrics$R2
      MAPE <- GrowthModel_OLS$fit_metrics$MAPE

      model <- "OLS"
    } else {
      y_pred <- GrowthModel_NLS$y_pred
      exp_gr <- GrowthModel_NLS$exp_gr
      y_0 <- GrowthModel_NLS$y_0
      R0 <- GrowthModel_NLS$R0
      doubling <- GrowthModel_NLS$doubling
      sum_of_squares <- GrowthModel_NLS$fit_metrics$SQT
      R2 <- GrowthModel_NLS$fit_metrics$R2
      MAPE <- GrowthModel_NLS$fit_metrics$MAPE

      model <- "NLS"
    }

    par(mar = c(5.1, 5, 4.1, 4.1))

    plot(
      t,
      y,
      col = cp_col,
      "p",
      pch = cp_pch,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    rect(
      par("usr")[1],
      par("usr")[3],
      par("usr")[2],
      par("usr")[4],
      col = bg_col
    )

    if (bgrid) {
      grid(
        col = bgrid_col,
        lty = bgrid_type,
        lwd = bgrid_size
      )
    }

    points(
      t,
      y,
      col = cp_col,
      pch = 19,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    lines(
      t,
      y_pred,
      col = cl_col
    )

    if (isTRUE(show_model_results)) {
      text(
        paste0("r = ", round(exp_gr, 5)),
        x = 0,
        y = max(y),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("y_0 = ", round(y_0, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.04 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("R0 = ", round(R0, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.12 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("doubling = ", round(doubling, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.16 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("SSQ = ", round(sum_of_squares, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.20 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("R^2 = ", round(R2, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.24 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("MAPE = ", round(MAPE, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.28 * anno_size)),
        pos = 4,
        cex = anno_size
      )
    }

    invisible(x)
  }
)

setMethod(
  "print",
  "expgrowth",
  function(x) {
    cat(model_descriptions[["expgrowth"]], "\n")
    cat("Use summary() for results", "\n")
  }
)

setMethod(
  "show",
  "expgrowth",
  function(object) {
    cat(model_descriptions[["expgrowth"]], "\n")
    cat("Use summary() for results", "\n")
  }
)


setClass(
  "loggrowth",
  slots = list(
    LinModel = "list",
    GrowthModel_OLS = "list",
    GrowthModel_NLS = "list",
    y = "numeric",
    t = "numeric",
    config = "list"
  )
)

setMethod(
  "summary",
  "loggrowth",
  function(object) {
    GrowthModel_OLS <- object@GrowthModel_OLS
    GrowthModel_NLS <- object@GrowthModel_NLS
    config <- object@config

    cat(model_descriptions[["loggrowth"]], "\n\n")

    cat("OLS model\n")
    cat(sprintf("  Saturation               : %.3f\n", GrowthModel_OLS$S))
    cat(sprintf("  Growth rate              : %.3f\n", GrowthModel_OLS$r))
    cat(sprintf("  Baseline                 : %.3f\n", GrowthModel_OLS$y_0))
    cat(sprintf("  Inflection point         : %.3f\n", GrowthModel_OLS$ip))
    cat(sprintf("  Time of inflection point : %.3f\n", GrowthModel_OLS$t_ip))
    cat(sprintf(
      "  R-squared                : %.3f\n\n",
      GrowthModel_OLS$fit_metrics$R2
    ))

    if (length(GrowthModel_NLS) > 0) {
      cat("NLS model\n")
      cat(sprintf("  Saturation               : %.3f\n", GrowthModel_NLS$S))
      cat(sprintf("  Growth rate              : %.3f\n", GrowthModel_NLS$r))
      cat(sprintf("  Baseline                 : %.3f\n", GrowthModel_NLS$y_0))
      cat(sprintf("  Inflection point         : %.3f\n", GrowthModel_NLS$ip))
      cat(sprintf("  Time of inflection point : %.3f\n", GrowthModel_NLS$t_ip))
      cat(sprintf(
        "  R-squared                : %.3f\n\n",
        GrowthModel_NLS$fit_metrics$R2
      ))
    }

    cat("Input\n")

    cat(sprintf(
      "  Time points (not NaN)    : %.0f\n",
      as.integer(object@config$TP)
    ))

    if (!config$nls) {
      cat("  NLS estimation         : Not desired by user\n")
    } else {
      if (isTRUE(config$nls_estimation)) {
        if (!is.null(config$S)) {
          cat(
            paste0("  Saturation estimation    : ", config$S_start_est_method),
            "\n"
          )
          cat(paste0("  Saturation value         : ", config$S), "\n")
        } else {
          cat(
            paste0("  Saturation estimation    : ", config$S_start_est_method),
            "\n"
          )
          cat(paste0("  Saturation start value   : ", config$S_start), "\n")
          cat(paste0("  Saturation end value     : ", config$S_end), "\n")
        }
      } else {
        cat("  NLS estimation         : Failed\n")
      }
    }

    invisible(object)
  }
)

setMethod(
  "plot",
  "loggrowth",
  function(
    x,
    y = NULL,
    cp_col = "lightblue",
    cp_pch = 19,
    cl_col = "red",
    plot_d = TRUE,
    dl_col = "blue",
    x_lab = "Time",
    y_lab = "Cumulative infections",
    y2_lab = "dC/dt",
    plot_title = "Logistic growth model",
    text_size = 1,
    anno_size = 0.7,
    bgrid = TRUE,
    bgrid_col = "white",
    bgrid_type = 1,
    bgrid_size = 1,
    bg_col = "white",
    show_model_results = TRUE
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    t <- x@t
    y <- x@y
    GrowthModel_NLS <- x@GrowthModel_NLS
    GrowthModel_OLS <- x@GrowthModel_OLS

    if (is.null(GrowthModel_NLS)) {
      y_pred <- GrowthModel_OLS$y_pred
      dy_dt <- GrowthModel_OLS$dy_dt
      r <- GrowthModel_OLS$r
      y_0 <- GrowthModel_OLS$y_0
      S <- GrowthModel_OLS$S
      ip <- GrowthModel_OLS$ip
      t_ip <- GrowthModel_OLS$t_ip
      sum_of_squares <- GrowthModel_OLS$fit_metrics$SQT
      R2 <- GrowthModel_OLS$fit_metrics$R2
      MAPE <- GrowthModel_OLS$fit_metrics$MAPE

      model <- "OLS"
    } else {
      y_pred <- GrowthModel_NLS$y_pred
      dy_dt <- GrowthModel_NLS$dy_dt
      r <- GrowthModel_NLS$r
      y_0 <- GrowthModel_NLS$y_0
      S <- GrowthModel_NLS$S
      ip <- GrowthModel_NLS$ip
      t_ip <- GrowthModel_NLS$t_ip
      sum_of_squares <- GrowthModel_NLS$fit_metrics$SQT
      R2 <- GrowthModel_NLS$fit_metrics$R2
      MAPE <- GrowthModel_NLS$fit_metrics$MAPE

      model <- "NLS"
    }

    par(mar = c(5.1, 5, 4.1, 4.1))

    plot(
      t,
      y,
      col = cp_col,
      "p",
      pch = cp_pch,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    rect(
      par("usr")[1],
      par("usr")[3],
      par("usr")[2],
      par("usr")[4],
      col = bg_col
    )

    if (bgrid) {
      grid(
        col = bgrid_col,
        lty = bgrid_type,
        lwd = bgrid_size
      )
    }

    points(
      t,
      y,
      col = cp_col,
      pch = 19,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    lines(
      t,
      y_pred,
      col = cl_col
    )

    if (isTRUE(show_model_results)) {
      text(
        paste0("r = ", round(r, 5)),
        x = 0,
        y = max(y),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("y_0 = ", round(y_0, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.04 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("S = ", round(S, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.08 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("ip = ", round(ip, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.12 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("t_ip = ", round(t_ip, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.16 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("SSQ = ", round(sum_of_squares, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.20 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("R^2 = ", round(R2, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.24 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("MAPE = ", round(MAPE, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.28 * anno_size)),
        pos = 4,
        cex = anno_size
      )
    }

    if (isTRUE(plot_d)) {
      par(new = TRUE)

      plot(
        t,
        dy_dt,
        xaxt = "n",
        yaxt = "n",
        ylab = "",
        xlab = "",
        col = dl_col,
        type = "l",
        cex.axis = text_size,
        cex.lab = text_size
      )

      axis(side = 4, cex.axis = text_size)
      mtext(y2_lab, side = 4, line = 3, cex = text_size)
    }

    invisible(x)
  }
)

setMethod(
  "print",
  "loggrowth",
  function(x) {
    cat(model_descriptions[["loggrowth"]], "\n")
    cat("Use summary() for results", "\n")

    invisible(x)
  }
)

setMethod(
  "show",
  "loggrowth",
  function(object) {
    cat(model_descriptions[["loggrowth"]], "\n")
    cat("Use summary() for results", "\n")

    invisible(object)
  }
)


setClass(
  "hawkes",
  slots = list(
    y = "numeric",
    t = "numeric",
    mu = "numeric",
    alpha = "numeric",
    beta = "numeric",
    br = "numeric",
    y_pred = "numeric",
    fit_metrics = "list",
    config = "list"
  )
)

setMethod(
  "summary",
  "hawkes",
  function(object) {
    cat(model_descriptions[["hawkes"]], "\n\n")

    mu <- object@mu
    alpha <- object@alpha
    beta <- object@beta
    br <- object@br
    fit_metrics <- object@fit_metrics
    config <- object@config

    cat("Model optimization\n")
    cat(sprintf("  Mu                       : %.3f\n", mu))
    cat(sprintf("  Alpha                    : %.3f\n", alpha))
    cat(sprintf("  Beta                     : %.3f\n", beta))
    cat(sprintf("  Breaking ratio           : %.3f\n\n", br))

    cat(sprintf("  Sum of squares           : %.3f\n", fit_metrics$SQR))
    cat(sprintf("  R-squared                : %.3f\n", fit_metrics$R2))
    cat(sprintf("  MAPE                     : %.3f\n\n", fit_metrics$MAPE))

    cat(sprintf("  Time points (not NaN)    : %.0f\n", as.integer(config$TP)))
    cat(
      paste0(
        "  Optimization method      : ",
        as.character(config$optim_method)
      ),
      "\n"
    )
  }
)

setMethod(
  "plot",
  "hawkes",
  function(
    x,
    y = NULL,
    cp_col = "lightblue",
    cp_pch = 19,
    cl_col = "red",
    x_lab = "Time",
    y_lab = "Daily infections",
    plot_title = "Hawkes process model",
    text_size = 1,
    anno_size = 0.7,
    bgrid = TRUE,
    bgrid_col = "white",
    bgrid_type = 1,
    bgrid_size = 1,
    bg_col = "white",
    show_model_results = TRUE
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    t <- x@t
    y <- x@y
    y_pred <- x@y_pred

    mu <- x@mu
    alpha <- x@alpha
    beta <- x@beta
    br <- x@br

    fit_metrics <- x@fit_metrics
    sum_of_squares <- fit_metrics$SQR
    R2 <- fit_metrics$R2
    MAPE <- fit_metrics$MAPE

    config <- x@config

    par(mar = c(5.1, 5, 4.1, 4.1))

    plot(
      t,
      y,
      col = cp_col,
      "p",
      pch = cp_pch,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    rect(
      par("usr")[1],
      par("usr")[3],
      par("usr")[2],
      par("usr")[4],
      col = bg_col
    )

    if (bgrid) {
      grid(
        col = bgrid_col,
        lty = bgrid_type,
        lwd = bgrid_size
      )
    }

    points(
      t,
      y,
      col = cp_col,
      pch = 19,
      xlab = x_lab,
      ylab = y_lab,
      main = plot_title,
      cex.axis = text_size,
      cex.lab = text_size,
      cex.main = text_size
    )

    lines(
      t,
      y_pred,
      col = cl_col
    )

    if (isTRUE(show_model_results)) {
      text(
        paste0("Mu = ", round(mu, 5)),
        x = 0,
        y = max(y),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("Alpha = ", round(alpha, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.04 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("Beta = ", round(beta, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.08 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("br = ", round(br, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.12 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("SSQ = ", round(sum_of_squares, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.20 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("R^2 = ", round(R2, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.24 * anno_size)),
        pos = 4,
        cex = anno_size
      )
      text(
        paste0("MAPE = ", round(MAPE, 3)),
        x = 0,
        y = (max(y) - (max(y) * 0.28 * anno_size)),
        pos = 4,
        cex = anno_size
      )
    }

    invisible(x)
  }
)

setMethod(
  "print",
  "hawkes",
  function(x) {
    cat(model_descriptions[["hawkes"]], "\n")
    cat("Use summary() for results", "\n")
  }
)

setMethod(
  "show",
  "hawkes",
  function(object) {
    cat(model_descriptions[["hawkes"]], "\n")
    cat("Use summary() for results", "\n")
  }
)


setClass(
  "breaksgrowth",
  slots = list(
    GrowthModel_OLS = "list",
    y = "numeric",
    t = "numeric",
    config = "list"
  )
)

setMethod(
  "summary",
  "breaksgrowth",
  function(object) {
    GrowthModel_OLS <- object@GrowthModel_OLS
    config <- object@config

    cat("Time Series Model with Breakpoints for Infections\n\n")

    cat("OLS model\n")

    cat(sprintf(
      "  No. of breakpoints       : %.0f\n",
      GrowthModel_OLS$bp_obj_breakpoints_no
    ))
    cat(sprintf(
      "  No. of segments          : %.0f\n",
      GrowthModel_OLS$bp_obj_segments_no
    ))
    cat(
      paste0(
        "  Breakpoints at positions : ",
        paste(GrowthModel_OLS$bp_obj_breakpoints, collapse = ", ")
      ),
      "\n"
    )

    cat("Input\n")

    cat(sprintf(
      "  Time points (not NaN)    : %.0f\n",
      as.integer(object@config$TP)
    ))
    if (isTRUE(object@config$ln)) {
      cat("  Natural logarithm        : YES", "\n")
    } else {
      cat("  Natural logarithm        : NO", "\n")
    }

    invisible(object)
  }
)

setMethod(
  "print",
  "breaksgrowth",
  function(x) {
    cat(paste0("Time Series Model with Breakpoints for Infections"), "\n")
    cat("Use summary() for results", "\n")
  }
)

setMethod(
  "show",
  "breaksgrowth",
  function(object) {
    cat(paste0("Time Series Model with Breakpoints for Infections"), "\n")
    cat("Use summary() for results", "\n")
  }
)

setMethod(
  "plot",
  "breaksgrowth",
  function(
    x,
    y = NULL,
    line.col = "chocolate",
    ci.col = c(
      rgb(0, 0, 255, maxColorValue = 255, alpha = 0.5),
      rgb(0, 255, 0, maxColorValue = 255, alpha = 0.5),
      rgb(255, 0, 0, maxColorValue = 255, alpha = 0.5)
    ),
    legend.show = TRUE,
    legend.pos = c(
      "topright",
      "topright",
      "topright",
      "topright"
    ),
    xlab = "Time",
    ylab = "Y",
    ylim = NULL,
    plot.main = c(
      "Breakpoints",
      "Segment model fits",
      "BIC and Residual Sum of Squares",
      "Model without breaks (one segment)"
    ),
    output.full = TRUE,
    separate_plots = FALSE,
    show_model_results = TRUE,
    ...
  ) {
    par_old <- par(no.readonly = TRUE)
    on.exit(par(par_old))
    dev.new()

    config <- x@config
    ln <- config$ln
    TP <- config$TP
    add_constant <- config$add_constant
    alpha <- config$alpha

    segments_models_df <- x@GrowthModel_OLS$segments_models_df
    segment_model_ci_points <- x@GrowthModel_OLS$segment_model_ci_points
    bp_obj_coef <- x@GrowthModel_OLS$bp_obj_coef
    bp_obj_breakpoints <- x@GrowthModel_OLS$bp_obj_breakpoints
    bp_obj_breakpoints_no <- x@GrowthModel_OLS$bp_obj_breakpoints_no
    bp_obj_segments_no <- x@GrowthModel_OLS$bp_obj_segments_no
    bp_obj_breakpoints_ci <- x@GrowthModel_OLS$bp_obj_breakpoints_ci
    bp_obj <- x@GrowthModel_OLS$bp_obj

    y <- as.numeric(x@y)
    t <- as.numeric(x@t)
    dataset <- data.frame(y, t)

    if (isTRUE(ln)) {
      bp_formula <- as.formula("log(y) ~ t")
    } else {
      bp_formula <- as.formula("y ~ t")
    }
    model_no_bp <- lm(
      bp_formula,
      data = dataset
    )

    ci_lower <- round((alpha / 2) * 100, 1)
    ci_upper <- round((1 - (alpha / 2)) * 100, 1)
    CI <- paste0("CI (", ci_lower, ";", ci_upper, ")")

    if ((output.full) & (!separate_plots)) {
      par(mfrow = c(2, 2))
    }

    col_ci <- ci.col[1]

    if (is.null(ylim)) {
      plot(
        model_no_bp$model[order(model_no_bp$model[, 2]), ][, 2],
        model_no_bp$model[order(model_no_bp$model[, 2]), ][, 1],
        lwd = 1.5,
        type = "l",
        col = line.col,
        xlab = xlab,
        ylab = ylab,
        main = plot.main[1]
      )
    } else {
      plot(
        model_no_bp$model[order(model_no_bp$model[, 2]), ][, 2],
        model_no_bp$model[order(model_no_bp$model[, 2]), ][, 1],
        lwd = 1.5,
        type = "l",
        col = line.col,
        xlab = xlab,
        ylab = ylab,
        main = plot.main[1],
        ylim = ylim
      )
    }

    if (bp_obj_breakpoints_no > 0) {
      i <- 0

      for (i in 1:bp_obj_breakpoints_no) {
        rect(
          bp_obj_breakpoints_ci[[1]][i],
          par("usr")[3],
          bp_obj_breakpoints_ci[[1]][
            i + ((2 / 3) * length(bp_obj_breakpoints_ci[[1]]))
          ],
          par("usr")[4],
          col = col_ci,
          lty = 0
        )

        abline(
          v = bp_obj_breakpoints_ci[[1]][
            i + ((1 / 3) * length(bp_obj_breakpoints_ci[[1]]))
          ],
          col = "blue",
          lty = 1,
          lwd = 1.5
        )
      }

      i <- 0

      for (i in 1:bp_obj_segments_no) {
        text(
          x = mean(c(segments_models_df[i, 1], segments_models_df[i, 2])),
          y = min(model_no_bp$model[, 1]),
          (round(segments_models_df[i, 7], 3)),
          cex = 1,
          col = "black"
        )
      }

      if (legend.show) {
        legend(
          legend.pos[1],
          legend = c("Breakpoint", CI),
          col = c("blue", col_ci),
          lty = c(1, NA),
          pch = c(NA, 15),
          lwd = 1.5,
          bty = "n",
          bg = "white"
        )

        text(
          min(model_no_bp$model[, 2]) - max(model_no_bp$model[, 2]) * 0.1,
          min(model_no_bp$model[, 1]),
          "Slope",
          xpd = NA
        )
      }
    }

    if (output.full) {
      col_ci <- ci.col[2]

      if (is.null(ylim)) {
        plot(
          model_no_bp$model[, 2],
          model_no_bp$model[, 1],
          pch = 19,
          col = line.col,
          cex = 0.8,
          xlab = xlab,
          ylab = ylab,
          main = plot.main[2]
        )
      } else {
        plot(
          model_no_bp$model[, 2],
          model_no_bp$model[, 1],
          pch = 19,
          col = line.col,
          cex = 0.8,
          xlab = xlab,
          ylab = ylab,
          main = plot.main[2],
          ylim = ylim
        )
      }

      i <- 0

      for (i in 1:bp_obj_segments_no) {
        pol_coords <-
          data.frame(
            cbind(
              model_no_bp$model[
                (segments_models_df[i, 1]):(segments_models_df[i, 2]),
                2
              ],
              rev(model_no_bp$model[
                (segments_models_df[i, 1]):(segments_models_df[i, 2]),
                2
              ]),
              segment_model_ci_points[[i]][, 2]
            ),
            rev(segment_model_ci_points[[i]][, 3])
          )
        colnames(pol_coords) <- c("x1", "x2", "y1", "y2")

        polygon(
          c(pol_coords$x1, pol_coords$x2),
          c(pol_coords$y1, pol_coords$y2),
          col = col_ci,
          border = NA
        )

        lines(
          x = (model_no_bp$model[
            (segments_models_df[i, 1]):(segments_models_df[i, 2]),
            2
          ]),
          y = (segment_model_ci_points[[i]][, 1]),
          col = "darkgreen",
          lty = 2,
          lwd = 1.5
        )

        text(
          x = mean(c(segments_models_df[i, 1], segments_models_df[i, 2])),
          y = min(model_no_bp$model[, 1]),
          (round(segments_models_df[i, 14], 3)),
          cex = 1,
          col = "black"
        )
      }

      if (legend.show) {
        legend(
          legend.pos[2],
          legend = c("Model fit", CI),
          col = c("darkgreen", col_ci),
          lty = c(2, NA),
          pch = c(NA, 15),
          lwd = 1.5,
          bty = "n",
          bg = "white"
        )

        text(
          min(model_no_bp$model[, 2]) - max(model_no_bp$model[, 2]) * 0.1,
          min(model_no_bp$model[, 1]),
          expression(R^2),
          xpd = NA
        )
      }

      plot(
        bp_obj,
        lwd = 1.5,
        main = plot.main[3]
      )

      if (is.null(ylim)) {
        plot(
          model_no_bp$model[, 2],
          model_no_bp$model[, 1],
          pch = 19,
          col = line.col,
          cex = 0.8,
          xlab = xlab,
          ylab = ylab,
          main = plot.main[4]
        )
      } else {
        plot(
          model_no_bp$model[, 2],
          model_no_bp$model[, 1],
          pch = 19,
          col = line.col,
          cex = 0.8,
          xlab = xlab,
          ylab = ylab,
          main = plot.main[4],
          ylim = ylim
        )
      }

      model_no_bp_ci_points <-
        predict(
          model_no_bp,
          interval = "confidence",
          level = 1 - alpha
        )

      col_ci <- ci.col[3]

      pol_coords <-
        data.frame(
          cbind(
            model_no_bp$model[, 2],
            rev(model_no_bp$model[, 2]),
            model_no_bp_ci_points[, 2],
            rev(model_no_bp_ci_points[, 3])
          )
        )

      colnames(pol_coords) <- c("x1", "x2", "y1", "y2")

      polygon(
        c(pol_coords$x1, pol_coords$x2),
        c(pol_coords$y1, pol_coords$y2),
        col = col_ci,
        border = NA
      )

      lines(
        x = model_no_bp$model[, 2],
        y = model_no_bp$fitted.values,
        lwd = 1.5,
        lty = 2,
        col = "red"
      )

      if (legend.show) {
        legend(
          legend.pos[4],
          legend = c("Model fit", CI),
          col = c("red", col_ci),
          lty = c(2, NA),
          pch = c(NA, 15),
          lwd = 1.5,
          bty = "n",
          bg = "white"
        )
      }
    }

    invisible(x)
  }
)


setGeneric(
  "add_timestamp",
  function(
    object,
    function_or_method = "",
    process = "",
    status = "OK"
  ) {
    standardGeneric("add_timestamp")
  }
)

setMethod(
  "add_timestamp",
  "ANY",
  function(
    object,
    function_or_method = "",
    process = "",
    status = "OK"
  ) {
    timestamp <- c(
      time = as.character(Sys.time()),
      package = paste0(package_name, " ", package_version),
      function_or_method = function_or_method,
      process = process,
      status = status
    )

    object@timestamp <- append(object@timestamp, list(timestamp))

    validObject(object)

    object
  }
)

setGeneric(
  "timestamps",
  function(
    object
  ) {
    standardGeneric("timestamps")
  }
)

setMethod(
  "timestamps",
  "ANY",
  function(
    object
  ) {
    timestamp <- object@timestamp

    for (i in seq_along(timestamp)) {
      entry <- timestamp[[i]]

      cat(
        paste0(
          i,
          " ",
          entry[["time"]],
          " | ",
          entry[["package"]],
          " | ",
          entry[["function_or_method"]],
          " | ",
          entry[["process"]],
          " | ",
          entry[["status"]]
        ),
        "\n"
      )
    }
  }
)


# Loading infections panel data (data.frame)
# and creating an object of class infpan:
load_infections_paneldata <-
  function(
    data,
    col_cases,
    col_date,
    col_region,
    other_cols = NULL,
    time_format = "%Y-%m-%d",
    time_unit = "days",
    verbose = FALSE
  ) {
    col_errors <- character(0)
    if (!col_cases %in% colnames(data)) {
      col_errors <- c(col_errors, col_cases)
    }
    if (!col_date %in% colnames(data)) {
      col_errors <- c(col_errors, col_date)
    }
    if (!col_region %in% colnames(data)) {
      col_errors <- c(col_errors, col_region)
    }
    if (length(col_errors) > 0) {
      stop(paste0(
        "Import failed. Columns not in data.frame: ",
        paste(col_errors, collapse = ", ")
      ))
    }

    data[[col_date]] <- as.Date(data[[col_date]], format = time_format)

    data <- data[order(data[[col_region]], data[[col_date]]), ]

    N <- nlevels(as.factor(data[[col_region]]))
    N_names <- levels(as.factor(data[[col_region]]))
    N_withoutcases <- 0
    TP <- nlevels(as.factor(data[[col_date]]))
    TP_t <- levels(as.factor(data[[col_date]]))

    data_check_balanced <-
      is_balanced(
        data,
        col_cases = col_cases,
        col_region = col_region,
        col_date = col_date
      )
    data_balanced <- data_check_balanced$data_balanced

    data_statistics <- c(N, TP, N_withoutcases, data_balanced)

    if (isTRUE(verbose)) {
      cat(paste0(
        "Infections panel data include ",
        N,
        " regions in column '",
        col_region,
        "' and ",
        TP,
        " time points in column '",
        col_date,
        "', with cases in column '",
        col_cases,
        "'. "
      ))

      if (isTRUE(data_balanced)) {
        cat("The data is balanced.")
      } else {
        cat("The data is not balanced.")
      }
      cat("\n")
    }

    if (!is.null(other_cols)) {
      for (name in names(other_cols)) {
        if (name %in% permitted_other_cols) {
          other_cols <- c(other_cols, name)
        } else {
          warning(
            paste0(
              "Column identifier '",
              name,
              "' is unknown. Permitted identifier for other columns are: ",
              paste(permitted_other_cols, collapse = ", "),
              "."
            ),
            "\n"
          )
        }
      }
    } else {
      other_cols <- character(0)
    }

    data[paste0(col_region, "_", col_date)] <-
      paste0(
        as.character(data[[col_region]]),
        "_",
        as.character(data[[col_date]])
      )

    infpan_object <-
      new(
        "infpan",
        input_data = data.frame(data),
        data_statistics = data_statistics,
        index_col_names = c(col_region, col_date),
        cases_col_name = col_cases,
        other_cols = other_cols,
        time_format = time_format,
        time_unit = time_unit
      )

    infpan_object <- add_timestamp(
      infpan_object,
      function_or_method = "load_infections_paneldata",
      process = "Import of infections panel data and creation of infpan object"
    )

    return(infpan_object)
  }


# Function for Swash-Backwash Model:
swash_backwash <-
  function(
    infpan = NULL,
    data = NULL,
    col_cases = NULL,
    col_date = NULL,
    col_region = NULL,
    time_format = "%Y-%m-%d",
    verbose = FALSE
  ) {
    if (is.null(infpan) && is.null(data)) {
      stop("Either an infpan object or a data.frame must be stated")
    }
    if (
      !is.null(data) &&
        (is.null(col_cases) || is.null(col_date) || is.null(col_region))
    ) {
      stop(
        "When specifying a data.frame, three parameters must be stated: 'col_cases', 'col_date', 'col_region'"
      )
    }
    if (!is.null(infpan) && !is.null(data)) {
      message(
        "Only a data.frame OR an infpan object must be stated. Parameter 'data' is ignored"
      )
      data <- NULL
    }

    if (!is.null(data)) {
      col_errors <- character(0)
      if (!col_cases %in% colnames(data)) {
        col_errors <- c(col_errors, col_cases)
      }
      if (!col_date %in% colnames(data)) {
        col_errors <- c(col_errors, col_date)
      }
      if (!col_region %in% colnames(data)) {
        col_errors <- c(col_errors, col_region)
      }
      if (length(col_errors) > 0) {
        stop(paste0(
          "Import failed. Columns not in data.frame: ",
          paste(col_errors, collapse = ", ")
        ))
      }

      if (isTRUE(verbose)) {
        cat("Reading and diagnosing infections panel data ... ")
      }

      N <- nlevels(as.factor(data[[col_region]]))
      N_names <- levels(as.factor(data[[col_region]]))
      N_withoutcases <- 0
      TP <- nlevels(as.factor(data[[col_date]]))
      TP_t <- levels(as.factor(data[[col_date]]))

      data_check_balanced <-
        is_balanced(
          data,
          col_cases = col_cases,
          col_region = col_region,
          col_date = col_date
        )
      data_balanced <- data_check_balanced$data_balanced

      if (isTRUE(verbose)) {
        cat("OK", "\n")
      }
    }

    if (!is.null(infpan)) {
      if (isTRUE(verbose)) {
        cat("Loading infections panel data object ... ")
      }

      data <- infpan@input_data

      col_region <- infpan@index_col_names[1]
      col_date <- infpan@index_col_names[2]
      col_cases <- infpan@cases_col_name

      time_format <- infpan@time_format

      N <- infpan@data_statistics[1]
      N_names <- levels(as.factor(data[[col_region]]))
      N_withoutcases <- infpan@data_statistics[3]
      TP <- infpan@data_statistics[2]
      TP_t <- levels(as.factor(data[[col_date]]))
      data_balanced <- infpan@data_statistics[4]

      if (isTRUE(verbose)) {
        cat("OK", "\n")
      }
    }

    data[[col_date]] <-
      as.Date(
        data[[col_date]],
        format = time_format
      )

    data <- data[order(data[[col_region]], data[[col_date]]), ]

    if (isTRUE(verbose)) {
      cat(paste0(
        "Infections panel data include ",
        N,
        " regions in column '",
        col_region,
        "' and ",
        TP,
        " time points in column '",
        col_date,
        "', with cases in column '",
        col_cases,
        "'. "
      ))

      if (isTRUE(data_balanced)) {
        cat("The data is balanced.")
      } else {
        cat("The data is not balanced.")
      }
      cat("\n")
    }

    if (isTRUE(verbose)) {
      cat(paste0(
        "Calculating Swash-Backwash Model for ",
        N,
        " regions and ",
        TP,
        " time points ... "
      ))
    }

    first_occ_regions <- data.frame(matrix(ncol = N + 1, nrow = TP))
    first_occ_regions[, 1] <- TP_t
    colnames(first_occ_regions)[1] <- "date"
    colnames(first_occ_regions)[2:(N + 1)] <- N_names

    last_occ_regions <- data.frame(matrix(ncol = N + 1, nrow = TP))
    last_occ_regions[, 1] <- TP_t
    colnames(last_occ_regions)[1] <- "date"
    colnames(last_occ_regions)[2:(N + 1)] <- N_names

    i <- 0

    for (i in 1:N) {
      data_n <- data[data[[col_region]] == N_names[i], ]
      data_n$occurence <- 0

      if (sum(data_n[[col_cases]], na.rm = TRUE) > 0) {
        data_n[data_n[[col_cases]] > 0, ]$occurence <- 1
      } else {
        N_withoutcases <- N_withoutcases + 1
      }

      first_occ <- which(data_n$occurence == 1)[1]
      data_n$LE <- 0
      data_n$LE[first_occ] <- 1

      first_occ_regions[, i + 1] <- data_n$LE

      colnames(first_occ_regions)[i + 1] <- paste0("Region_", N_names[i])

      last_occ <- which(data_n$occurence == 1)[length(which(
        data_n$occurence == 1
      ))]
      data_n$FE <- 0
      data_n$FE[last_occ] <- 1

      last_occ_regions[, i + 1] <- data_n$FE

      colnames(last_occ_regions)[i + 1] <- paste0("Region_", N_names[i])
    }

    first_occ_regions$no_regions_LE <- rowSums(first_occ_regions[, 2:(N + 1)])
    first_occ_regions$t <- seq(1:TP)
    first_occ_regions$t_x_nt <- first_occ_regions$t *
      first_occ_regions$no_regions_LE

    t_LE <- sum(first_occ_regions$t_x_nt) / N

    last_occ_regions$no_regions_FE <- rowSums(last_occ_regions[, 2:(N + 1)])
    last_occ_regions$t <- seq(1:TP)
    last_occ_regions$t_x_nt <- last_occ_regions$t *
      last_occ_regions$no_regions_FE

    t_FE <- sum(last_occ_regions$t_x_nt) / N

    S_A <- (t_LE - 1) / TP
    I_A <- (t_FE / TP) - S_A
    R_A <- 1 - (S_A + I_A)
    R_0A <- (1 - S_A) / (1 - R_A)

    integrals <-
      c(
        S_A = S_A,
        I_A = I_A,
        R_A = R_A
      )

    velocity <-
      c(
        t_LE = t_LE,
        t_FE = t_FE,
        diff = t_FE - t_LE
      )

    occ_regions <- data.frame(
      first_occ_regions$date,
      first_occ_regions[, N + 2],
      last_occ_regions[, N + 2]
    )
    colnames(occ_regions) <- c("date", "LE", "FE")
    occ_regions$LE_FE <- occ_regions$LE - occ_regions$FE

    SIR_regions <- data.frame(matrix(ncol = 4, nrow = nrow(occ_regions)))
    SIR_regions[, 1] <- occ_regions$date
    SIR_regions[, 2] <- N - cumsum(occ_regions$LE)
    SIR_regions[, 3] <- cumsum(occ_regions$LE_FE)
    SIR_regions[, 4] <- cumsum(occ_regions$FE)
    colnames(SIR_regions) <-
      c(
        "date",
        "susceptible",
        "infected",
        "recovered"
      )

    cases_by_date <- aggregate(
      data[[col_cases]],
      by = list(data[[col_date]]),
      FUN = sum
    )
    colnames(cases_by_date) <- c("date", "cases")

    cases_by_region <- aggregate(
      data[[col_cases]],
      by = list(data[[col_region]]),
      FUN = sum
    )
    colnames(cases_by_region) <- c("region", "cases_cumulative")

    data_statistics <-
      c(
        N,
        TP,
        N_withoutcases,
        data_balanced
      )

    col_names <- c(col_cases, col_date, col_region)

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    sbm_object <-
      new(
        "sbm",
        R_0A = R_0A,
        integrals = integrals,
        velocity = velocity,
        occ_regions = occ_regions,
        SIR_regions = SIR_regions,
        cases_by_date = cases_by_date,
        cases_by_region = cases_by_region,
        input_data = data.frame(data),
        data_statistics = data_statistics,
        col_names = col_names
      )

    sbm_object <- add_timestamp(
      sbm_object,
      function_or_method = "swash_backwash",
      process = paste0(
        "Calculation of Swash-Backwash Model for ",
        N,
        " regions and ",
        TP,
        " time points"
      )
    )

    invisible(sbm_object)
  }


compare_countries <-
  function(
    sbm1,
    sbm2,
    country_names = c("Country 1", "Country 2"),
    indicator = "R_0A",
    iterations = 20,
    samples_ratio = 0.8,
    alpha = 0.05,
    replace = TRUE
  ) {
    country_names <- as.character(country_names)

    sbm1_confint <-
      confint(
        sbm1,
        iterations = iterations,
        samples_ratio = samples_ratio,
        alpha = alpha,
        replace = replace
      )

    sbm2_confint <-
      confint(
        sbm2,
        iterations = iterations,
        samples_ratio = samples_ratio,
        alpha = alpha,
        replace = replace
      )

    D <- sbm1_confint@iterations[[indicator]] -
      sbm2_confint@iterations[[indicator]]

    D_ci <- quantile_ci(
      x = D,
      alpha = alpha
    )

    bootstrap_config <- list(
      iterations = iterations,
      samples_ratio = samples_ratio,
      alpha = alpha,
      replace = replace
    )

    new(
      "countries",
      sbm_ci1 = sbm1_confint,
      sbm_ci2 = sbm2_confint,
      D = D,
      D_ci = D_ci,
      config = bootstrap_config,
      country_names = country_names,
      indicator = indicator
    )
  }


exponential_growth <-
  function(
    y,
    t,
    GI = 4,
    nls = TRUE,
    nls_start = list(a = 1, b = 0.1),
    add_constant = 1,
    verbose = FALSE
  ) {
    if (!is.numeric(y)) {
      stop("Infections vector must be of class 'numeric'.")
    }

    if (!is.numeric(t) & !is.Date(t)) {
      stop("Time vector must be of class 'numeric' or 'Date'.")
    }

    if (length(y) != length(t)) {
      stop(paste0(
        "Vectors 'y' and 't' differ in length: ",
        length(y),
        ", ",
        length(t)
      ))
    }

    yt_df <- data.frame(y, t)
    yt_df_length_clean <- yt_df[complete.cases(yt_df), ]

    if (nrow(yt_df_length_clean) < nrow(yt_df)) {
      y <- yt_df$y
      t <- yt_df$t

      nan_cases <- nrow(yt_df) - nrow(yt_df_length_clean)

      warning(paste0(nan_cases, " NaN values were removed"))
    }

    TP <- length(y)

    class_t <- "numeric"

    if (is.Date(t)) {
      class_t <- "Date"

      message("Time vector is of class 'Date'. Calculating time counter.")

      start_date <- min(t)

      time_counter <- as.integer(t - start_date)

      t <- time_counter
    }

    if (any(y <= 0)) {
      if (!is.null(add_constant) && is.numeric(add_constant)) {
        y_lin <- y + add_constant

        message(paste0(
          "Input infections data contains values <= 0. All values are increased by a constant equal to ",
          add_constant,
          " for linear estimation."
        ))
      } else {
        y_lin <- y[y > 0]

        warning(paste0(
          "Input infections data contains values <= 0. These values are skipped for linear estimation."
        ))
      }
    } else {
      y_lin <- y
    }

    if (isTRUE(verbose)) {
      cat("Performing OLS estimation ... ")
    }

    log_y_lin <- log(y_lin)

    linexpmodel <- lm(log_y_lin ~ t)

    y_0 <- linexpmodel$coefficients[1]

    exp_gr <- linexpmodel$coefficients[2]

    R0 <- exp(exp_gr * GI)

    doubling <- log(2) / exp_gr

    y_pred <- exp(predict(linexpmodel))

    fit_metrics <- metrics(
      observed = y_lin,
      expected = y_pred,
      plot = FALSE
    )

    fit_metrics <- fit_metrics[[1]]

    model_growth_ols_list <-
      list(
        exp_gr = exp_gr,
        y_0 = y_0,
        R0 = R0,
        doubling = doubling,
        y_pred = y_pred,
        model_data = linexpmodel,
        fit_metrics = fit_metrics
      )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    model_growth_nls_list <- list()
    nls_estimation <- TRUE
    nls_error_message <- NULL

    if (isTRUE(nls)) {
      if (isTRUE(verbose)) {
        cat("Trying nonlinear estimation ... ")
      }

      tryCatch(
        {
          expmodel <-
            nls(
              y ~ a * exp(b * t),
              start = nls_start
            )

          expmodel_summary <- summary(expmodel)

          y_0_NLS <- expmodel_summary$coefficients[1]

          exp_gr_NLS <- expmodel_summary$coefficients[2]

          R0_NLS <- exp(exp_gr_NLS * GI)

          doubling_NLS <- log(2) / exp_gr_NLS

          y_pred_NLS <- predict(expmodel)

          fit_metrics_NLS <- metrics(
            observed = y,
            expected = y_pred_NLS,
            plot = FALSE
          )

          fit_metrics_NLS <- fit_metrics_NLS[[1]]

          model_growth_nls_list <-
            list(
              exp_gr = exp_gr_NLS,
              y_0 = y_0_NLS,
              R0 = R0_NLS,
              doubling = doubling_NLS,
              y_pred = y_pred_NLS,
              model_data = expmodel,
              fit_metrics = fit_metrics_NLS
            )
        },

        error = function(cond) {
          nls_error_message <<- paste0(
            "Nonlinear estimation failed: ",
            conditionMessage(cond)
          )
          message(nls_error_message)
        }
      )

      if (length(model_growth_nls_list) == 0) {
        nls_estimation <- FALSE
      }

      if (isTRUE(verbose)) {
        cat("OK", "\n")
      }
    }

    config <-
      list(
        TP = TP,
        GI = GI,
        nls = nls,
        nls_start = nls_start,
        nls_estimation = nls_estimation,
        nls_error_message = nls_error_message
      )

    new(
      "expgrowth",
      GrowthModel_OLS = model_growth_ols_list,
      GrowthModel_NLS = model_growth_nls_list,
      y = y,
      t = t,
      config = config
    )
  }


logistic_growth <-
  function(
    y,
    t,
    S = NULL,
    S_start = NULL,
    S_end = NULL,
    S_iterations = 10,
    S_start_est_method = "bisect",
    seq_by = 10,
    nls = TRUE,
    add_constant = 1,
    verbose = FALSE
  ) {
    loggrowth_saturation <-
      function(
        y,
        t,
        S_start,
        S_end,
        S_start_est_method = "bisect",
        S_iterations = 10,
        seq_by = 10,
        verbose = FALSE
      ) {
        if (isTRUE(verbose)) {
          cat(paste0(
            "Estimating saturation value via method: ",
            S_start_est_method,
            " ... "
          ))
        }

        if (S_start_est_method == "trialanderror") {
          values_seq <- seq(S_start, S_end, by = seq_by)
          values_no <- length(values_seq)
          values_ssq <- matrix(ncol = 2, nrow = values_no)
          values_ssq[, 1] <- values_seq

          i <- 0

          for (i in 1:values_no) {
            y_new <- log((1 / y) - (1 / values_seq[i]))
            model_lin <- lm(y_new ~ t)
            model_lin_summary <- summary(model_lin)
            sum_of_squares_lin <- sum(model_lin_summary$residuals^2)

            values_ssq[i, 2] <- sum_of_squares_lin
          }

          values_ssq_order <- values_ssq[order(values_ssq[, 2])]

          S_est <- values_ssq_order[1]

          plot(values_ssq[, 1], values_ssq[, 2], "l")
        } else {
          i <- 0

          interval_m <- vector()

          for (i in 1:S_iterations) {
            interval_m[i] <- (S_start + S_end) / 2

            y_new_start <- log((1 / y) - (1 / S_start))

            model_lin_start <- lm(y_new_start ~ t)
            model_lin_start_summary <- summary(model_lin_start)
            sum_of_squares_lin_start <- sum(model_lin_start_summary$residuals^2)

            y_new_m <- log((1 / y) - (1 / interval_m[i]))
            model_lin_m <- lm(y_new_m ~ t)
            model_lin_m_summary <- summary(model_lin_m)
            sum_of_squares_lin_m <- sum(model_lin_m_summary$residuals^2)

            y_new_end <- log((1 / y) - (1 / S_end))
            model_lin_end <- lm(y_new_m ~ t)
            model_lin_end_summary <- summary(model_lin_m)
            sum_of_squares_lin_end <- sum(model_lin_end_summary$residuals^2)

            if (sum_of_squares_lin_start < sum_of_squares_lin_end) {
              S_start <- S_start
              S_end <- interval_m[i]
            } else {
              S_start <- interval_m[i]
              S_end <- S_end
            }
          }
          S_est <- interval_m[i]
        }

        if (isTRUE(verbose)) {
          cat("OK", "\n")
        }

        return(S_est)
      }

    if (!is.numeric(y)) {
      stop("Cumulative infections vector must be of class 'numeric'.")
    }

    if (!is.numeric(t) & !is.Date(t)) {
      stop("Time vector must be of class 'numeric' or 'Date'.")
    }

    if (length(y) != length(t)) {
      stop(paste0(
        "Vectors 'y' and 't' differ in length: ",
        length(y),
        ", ",
        length(t)
      ))
    }

    yt_df <- data.frame(y, t)
    yt_df_length_clean <- yt_df[complete.cases(yt_df), ]

    if (nrow(yt_df_length_clean) < nrow(yt_df)) {
      y <- yt_df$y
      t <- yt_df$t

      nan_cases <- nrow(yt_df) - nrow(yt_df_length_clean)
      warning(paste0(nan_cases, " NaN values were removed"))
    }

    if (any(y <= 0)) {
      if (!is.null(add_constant) && is.numeric(add_constant)) {
        y_lin <- y + add_constant

        message(paste0(
          "Input infections data contains values <= 0. All values are increased by a constant equal to ",
          add_constant,
          " for linear estimation."
        ))
      } else {
        y_lin <- y[y > 0]

        warning(paste0(
          "Input infections data contains values <= 0. These values are skipped for linear estimation."
        ))
      }
    } else {
      y_lin <- y
    }

    TP <- length(y_lin)

    class_t <- "numeric"

    if (is.Date(t)) {
      class_t <- "Date"

      message("Time vector is of class 'Date'. Calculating time counter.")

      start_date <- min(t)

      time_counter <- as.integer(t - start_date)

      t <- time_counter
    }

    if (is.null(S) & (is.null(S_start) | is.null(S_end))) {
      stop(
        "Saturation value or start and end values are required for estimation"
      )
    }

    if (!is.null(S)) {
      if (any(y <= 0) && !is.null(add_constant)) {
        S <- S + (add_constant * TP)

        message(paste0(
          "Input infections data contains values <= 0 and constant is set to ",
          add_constant,
          ". Saturation is increased to ",
          S,
          " for linear estimation"
        ))
      }

      if (S < max(y_lin)) {
        stop(paste0(
          "Saturation value must be above or equal to the the maximum of y (",
          max(y_lin),
          ")"
        ))
      }
    }

    if ((!is.null(S_start) & (!is.null(S_end)))) {
      if (any(y <= 0) && !is.null(add_constant)) {
        S_start <- S_start + (add_constant * TP)
        S_end <- S_end + (add_constant * TP)

        message(paste0(
          "Input infections data contains values <= 0 and constant is set to ",
          add_constant,
          ". Saturation start and end values are increased to ",
          S_start,
          " and ",
          S_end,
          " for linear estimation"
        ))
      }

      if ((S_start < max(y_lin)) || (S_end < max(y_lin))) {
        stop(paste0(
          "Start and end values of saturation must be above or equal to the the maximum of y (",
          max(y_lin),
          ")"
        ))
      }

      S <-
        loggrowth_saturation(
          y = y_lin,
          t = t,
          S_start = S_start,
          S_end = S_end,
          S_iterations = S_iterations,
          S_start_est_method = S_start_est_method,
          seq_by = seq_by,
          verbose = verbose
        )
    }

    if (isTRUE(verbose)) {
      cat("Performing OLS estimation ... ")
    }

    y_new <- log((1 / y_lin) - (1 / S))

    model_lin <- lm(y_new ~ t)

    model_lin_summary <- summary(model_lin)

    b <- model_lin_summary$coefficients[1]
    m <- model_lin_summary$coefficients[2]

    sum_of_squares_lin <- sum(model_lin_summary$residuals^2)

    model_lin_ols_list <-
      list(
        b = b,
        m = m,
        sum_of_squares = sum_of_squares_lin
      )

    r <- -m / S

    y_0 <- S / (1 + S * exp(m * t[1] + b))

    ip <- S / 2
    c <- -log(y_0 / (S - y_0))
    t_ip <- c / (r * S)

    y_pred <- S / (1 + S * exp(m * t + b))

    dy_dt <- r * y_pred * (1 - (y_pred / S))

    d2y_dt2 <- r * dy_dt * (1 - ((2 * y_pred) / S))

    fit_metrics_ols <- metrics(
      observed = y_lin,
      expected = y_pred,
      plot = FALSE
    )

    fit_metrics_ols <- fit_metrics_ols[[1]]

    model_growth_ols_list <-
      list(
        S = S,
        r = r,
        y_0 = y_0,
        ip = ip,
        t_ip = t_ip,
        y_pred = y_pred,
        fit_metrics = fit_metrics_ols,
        dy_dt = dy_dt,
        d2y_dt2 = d2y_dt2
      )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    model_growth_nls_list <- list()
    nls_estimation <- TRUE
    nls_error_message <- NULL

    if (nls) {
      if (isTRUE(verbose)) {
        cat("Trying nonlinear estimation ... ")
      }

      tryCatch(
        {
          model_nls <-
            nls(
              y ~ y_0 * S / (y_0 + (S - y_0) * exp(-r * S * t)),
              start = list(y_0 = y_0, S = S, r = r),
              control = list(maxiter = 500)
            )

          y_0_nls <- model_nls$m$getPars()[1]

          S_nls <- model_nls$m$getPars()[2]

          r_nls <- model_nls$m$getPars()[3]

          ip_nls <- S_nls / 2
          c_nls <- -log(y_0_nls / (S_nls - y_0_nls))
          t_ip_nls <- c_nls / (r_nls * S_nls)

          y_pred <- model_nls$m$predict()

          dy_dt <- r_nls * y_pred * (1 - (y_pred / S_nls))

          d2y_dt2 <- r_nls * dy_dt * (1 - ((2 * y_pred) / S_nls))

          fit_metrics_nls <- metrics(
            observed = y,
            expected = y_pred,
            plot = FALSE
          )

          fit_metrics_nls <- fit_metrics_nls[[1]]

          model_growth_nls_list <-
            list(
              S = S_nls,
              r = r_nls,
              y_0 = y_0_nls,
              ip = ip_nls,
              t_ip = t_ip_nls,
              y_pred = y_pred,
              fit_metrics = fit_metrics_nls,
              dy_dt = dy_dt,
              d2y_dt2 = d2y_dt2
            )

          if (isTRUE(verbose)) {
            cat("OK", "\n")
          }
        },

        error = function(cond) {
          nls_error_message <<- paste0(
            "Nonlinear estimation failed: ",
            conditionMessage(cond)
          )
          message(nls_error_message)
        }
      )

      if (length(model_growth_nls_list) == 0) {
        nls_estimation <- FALSE
      }
    }

    config <-
      list(
        S = S,
        S_start = S_start,
        S_end = S_end,
        S_iterations = S_iterations,
        S_start_est_method = S_start_est_method,
        seq_by = seq_by,
        nls = nls,
        class_t = class_t,
        TP = TP,
        nls_estimation = nls_estimation,
        nls_error_message = nls_error_message
      )

    new(
      "loggrowth",
      LinModel = model_lin_ols_list,
      GrowthModel_OLS = model_growth_ols_list,
      GrowthModel_NLS = model_growth_nls_list,
      y = y,
      t = t,
      config = config
    )
  }

hawkes_growth <-
  function(
    y,
    optim_method = "L-BFGS-B",
    verbose = FALSE
  ) {
    if (!is.numeric(y)) {
      stop("Daily infections vector must be of class 'numeric'.")
    }

    TP <- length(y)

    loglik <-
      function(par) {
        mu <- par[1]
        alpha <- par[2]
        beta <- par[3]

        lambda <- sapply(1:TP, function(t) {
          mu + sum(alpha * exp(-beta * (t - 1:(t - 1))) * y[1:(t - 1)])
        })

        -sum(dpois(y, lambda, log = TRUE))
      }

    fit <-
      optim(
        c(
          mu = mean(y),
          alpha = 0.2,
          beta = 0.2
        ),
        loglik,
        method = optim_method,
        lower = c(
          1e-6,
          1e-6,
          1e-6
        )
      )

    mu <- fit$par["mu"]
    alpha <- fit$par["alpha"]
    beta <- fit$par["beta"]
    br <- alpha / beta

    y_pred <- numeric(TP)

    for (t in 1:TP) {
      if (t == 1) {
        y_pred[t] <- mu
      } else {
        y_pred[t] <- mu +
          sum(alpha * exp(-beta * (t - 1:(t - 1))) * y[1:(t - 1)])
      }
    }

    fit_metrics <- metrics(
      observed = y,
      expected = y_pred,
      plot = FALSE
    )

    fit_metrics <- fit_metrics[[1]]

    config <-
      list(
        TP = TP,
        optim_method = optim_method
      )

    hawkes_object <-
      new(
        "hawkes",
        y = y,
        t = 1:TP,
        mu = mu,
        alpha = alpha,
        beta = beta,
        br = br,
        y_pred = y_pred,
        fit_metrics = fit_metrics,
        config = config
      )
  }


nbmatrix <-
  function(
    polygon_sf,
    ID_col,
    row.names = NULL
  ) {
    nb <- spdep::poly2nb(
      polygon_sf,
      row.names = row.names
    )

    polygon_sf$ID <- rownames(polygon_sf)

    poly_no <- nrow(polygon_sf)

    polys_nb <- data.frame(matrix(ncol = 3))
    colnames(polys_nb) <- c("ID", "ID_nb", "nb")

    i <- 0

    for (i in 1:poly_no) {
      neighbors <- nb[[i]]

      poly_id <- polygon_sf$ID[i]

      poly_nb <- data.frame(rep(poly_id, length(neighbors)), neighbors)
      colnames(poly_nb) <- c("ID", "ID_nb")
      poly_nb$nb <- 1

      polys_nb <- rbind(polys_nb, poly_nb)
    }

    polys_nb <- polys_nb[!is.na(polys_nb$ID), ]

    polygon_sf_IDs <- cbind(polygon_sf$ID, polygon_sf[[ID_col]])
    colnames(polygon_sf_IDs) <- c("ID", "ID2")

    polys_nb <-
      merge(
        polys_nb,
        polygon_sf_IDs,
        by.x = "ID",
        by.y = "ID"
      )

    polys_nb <-
      merge(
        polys_nb,
        polygon_sf_IDs,
        by.x = "ID_nb",
        by.y = "ID"
      )

    polys_nb <- polys_nb[c(2, 4, 1, 5, 3)]

    colnames(polys_nb) <- c("ID", "ID2", "ID_nb", "ID2_nb", "nb")

    nbmat_results <-
      list(
        nb = nb,
        nbmat = polys_nb
      )

    return(nbmat_results)
  }


nbstat <-
  function(
    polygon_sf,
    ID_col,
    link_data,
    data_ID_col,
    data_col,
    func = "sum",
    row.names = NULL
  ) {
    nbmat <- nbmatrix(
      polygon_sf,
      ID_col,
      row.names = row.names
    )

    nbmat <- nbmat[[2]]

    nbmat_data <-
      merge(
        nbmat,
        link_data,
        by.x = "ID2_nb",
        by.y = data_ID_col
      )

    nbmat_data_aggregate <-
      aggregate(
        nbmat_data[[data_col]],
        by = list(nbmat_data$ID2),
        FUN = func,
        na.rm = TRUE
      )

    colnames(nbmat_data_aggregate) <- c("ID2", paste0(data_col, "_", func))

    nbstat_results <-
      list(
        nbmat = nbmat,
        nbmat_data = nbmat_data,
        nbmat_data_aggregate = nbmat_data_aggregate
      )

    return(nbstat_results)
  }


plot_coef_ci <- function(
  point_estimates,
  confint_lower,
  confint_upper,
  coef_names,
  p = NULL,
  estimate_colors = NULL,
  confint_colors = NULL,
  auto_color = FALSE,
  alpha = 0.05,
  set_estimate_colors = c("red", "grey", "green"),
  set_confint_colors = c("#ffcccb", "lightgray", "#CCFFCC"),
  skipvars = NULL,
  plot.xlab = "Independent variables",
  plot.main = "Point estimates with CI",
  axis.at = seq(-30, 40, by = 5),
  pch = 15,
  cex = 2,
  lwd = 5,
  y.cex = 0.8
) {
  if (
    length(coef_names) == length(point_estimates) &&
      length(point_estimates) == length(confint_lower) &&
      length(confint_lower) == length(confint_upper)
  ) {
    data_plot <-
      data.frame(
        coef_names,
        point_estimates,
        confint_lower,
        confint_upper
      )
  } else {
    stop(
      "Vectors coef_names, point_estimates, confint_lower, and confint_upper differ in length."
    )
  }

  if (!is.null(p)) {
    if (length(p) == nrow(data_plot)) {
      data_plot$p <- p
    } else {
      stop(
        "Vector p differs in length from coef_names, point_estimates, confint_lower, and confint_upper"
      )
    }
  }

  if (!is.null(estimate_colors) & !is.null(confint_colors)) {
    data_plot$col_point <- estimate_colors
    data_plot$col_confint <- confint_colors
  } else {
    if (is.null(p)) {
      if (!auto_color) {
        stop(
          "If estimate_colors and confint_colors are NULL and auto_color is FALSE, then p must be stated as numeric vector"
        )
      } else {
        data_plot$col_point <- set_estimate_colors[2]
        data_plot$col_confint <- set_confint_colors[2]

        data_plot[
          (data_plot$point_estimates < 0) &
            (data_plot$confint_lower < 0) &
            (data_plot$confint_upper < 0),
        ]$col_point <- set_estimate_colors[1]
        data_plot[
          (data_plot$point_estimates < 0) &
            (data_plot$confint_lower < 0) &
            (data_plot$confint_upper < 0),
        ]$col_confint <- set_confint_colors[1]

        data_plot[
          (data_plot$point_estimates > 0) &
            (data_plot$confint_lower > 0) &
            (data_plot$confint_upper > 0),
        ]$col_point <- set_estimate_colors[3]
        data_plot[
          (data_plot$point_estimates > 0) &
            (data_plot$confint_lower > 0) &
            (data_plot$confint_upper > 0),
        ]$col_confint <- set_confint_colors[3]
      }
    }
  }

  if (!is.null(skipvars)) {
    data_plot <- data_plot[!data_plot$coef_names %in% skipvars, ]
  }

  if (!is.null(p)) {
    data_plot$col_point <- set_estimate_colors[2]
    data_plot$col_confint <- set_confint_colors[2]

    if (
      nrow(data_plot[
        (data_plot$point_estimates < 0) & (data_plot$p < alpha),
      ]) >
        0
    ) {
      data_plot[
        (data_plot$point_estimates < 0) & (data_plot$p < alpha),
      ]$col_point <- set_estimate_colors[1]
    }
    if (
      nrow(data_plot[
        (data_plot$point_estimates > 0) & (data_plot$p < alpha),
      ]) >
        0
    ) {
      data_plot[
        (data_plot$point_estimates > 0) & (data_plot$p < alpha),
      ]$col_point <- set_estimate_colors[3]
    }

    if (nrow(data_plot[data_plot$col_point == set_estimate_colors[1], ]) > 0) {
      data_plot[
        data_plot$col_point == set_estimate_colors[1],
      ]$col_confint <- set_confint_colors[1]
    }
    if (nrow(data_plot[data_plot$col_point == set_estimate_colors[3], ]) > 0) {
      data_plot[
        data_plot$col_point == set_estimate_colors[3],
      ]$col_confint <- set_confint_colors[3]
    }
  }

  data_plot$order <- seq_len(nrow(data_plot))
  data_plot <- data_plot[order(-data_plot$order), ]

  par(mar = c(4, 15, 2, 1))

  plot(
    x = data_plot$point_estimates,
    y = (seq_len(nrow(data_plot))),
    xlim = c(
      -max(abs(data_plot$confint_lower)),
      max(abs(data_plot$confint_upper))
    ),
    yaxt = "n",
    ylab = "",
    xaxt = "n",
    cex = 0.1,
    xlab = plot.xlab,
    main = plot.main
  )

  axis(
    1,
    at = axis.at,
    tck = 1,
    lty = 2,
    col = "gray"
  )

  par(las = 1)

  axis(
    side = 2,
    at = seq_len(nrow(data_plot)),
    labels = data_plot$coef_names,
    cex.axis = y.cex,
    tick = FALSE
  )

  abline(h = 0)
  abline(v = 0)

  i <- 0

  for (i in seq_len(nrow(data_plot))) {
    lines(
      x = c(
        data_plot$confint_lower[i],
        data_plot$confint_upper[i]
      ),
      y = c(i, i),
      lwd = lwd,
      col = data_plot$col_confint[i]
    )
  }

  points(
    x = data_plot$point_estimates,
    y = seq_len(nrow(data_plot)),
    pch = pch,
    cex = cex,
    col = data_plot$col_point
  )
}


metrics <-
  function(
    observed,
    expected,
    plot = TRUE,
    plot.main = "Observed vs. expected",
    xlab = "Observed",
    ylab = "Expected",
    point.col = "blue",
    point.pch = 19,
    line.col = "red",
    plot_residuals.main = "Residuals",
    legend.cex = 0.7
  ) {
    if (length(observed) != length(expected)) {
      stop("Vectors 'observed' and 'expected' differ in length")
    }

    observed_expected <- data.frame(observed, expected)

    observed_expected$residuals <- observed_expected$expected -
      observed_expected$observed
    observed_expected$residuals_abs <- abs(
      observed_expected$expected - observed_expected$observed
    )
    observed_expected$residuals_sq <- (observed_expected$expected -
      observed_expected$observed)^2
    observed_expected$residuals_rel <- observed_expected$residuals /
      observed_expected$observed *
      100
    observed_expected$residuals_rel_abs <- abs(observed_expected$residuals_rel)

    SQR <- sum(observed_expected$residuals_sq)
    SAR <- sum(observed_expected$residuals_abs)
    SQT <- sum(
      (observed_expected$observed - mean(observed_expected$observed))^2
    )
    R2 <- (1 - (SQR / SQT))
    MSE <- mean((observed_expected$observed - observed_expected$expected)^2)
    RMSE <- sqrt(MSE)
    MAE <- sum(observed_expected$observed - observed_expected$expected)
    MAPE <- mean(observed_expected$residuals_rel_abs)

    if (plot) {
      min <- min(observed_expected$observed)
      max <- max(observed_expected$observed)

      plot(
        observed_expected$observed,
        observed_expected$expected,
        xlab = xlab,
        ylab = ylab,
        pch = point.pch,
        col = point.col,
        main = plot.main,
        xlim = c((min - min * 0.1), (max * 1.1)),
        ylim = c((min - min * 0.1), (max * 1.1))
      )

      legend(
        "topleft",
        legend = c(
          bquote(R^2 == .(round(R2, 2))),
          paste0("MSE = ", round(MSE, 2)),
          paste0("RMSE = ", round(RMSE, 2)),
          paste0("MAE = ", round(MAE, 2)),
          paste0("MAPE = ", round(MAPE, 2))
        ),
        cex = legend.cex
      )

      abline(coef = c(0, 1), col = line.col)

      Y_residuals_stat <-
        data.frame(
          table(
            cut(
              observed_expected$residuals_rel,
              breaks = c(-Inf, seq(-90, 90, 10), Inf)
            )
          )
        )
      colnames(Y_residuals_stat) <- c("Range", "Freq")

      Y_residuals_stat$Freq_rel <- round(
        Y_residuals_stat$Freq / sum(Y_residuals_stat$Freq) * 100,
        2
      )

      barplot(
        Y_residuals_stat$Freq_rel,
        names.arg = Y_residuals_stat$Range,
        ylim = c(0, max(Y_residuals_stat$Freq_rel) * 1.5),
        main = plot_residuals.main
      )

      legend(
        "topleft",
        legend = c(
          paste0(
            "+/- 30 %: ",
            round(sum(Y_residuals_stat[8:13, ]$Freq_rel), 1),
            " %"
          ),
          paste0(
            "+/- 20 %: ",
            round(sum(Y_residuals_stat[9:12, ]$Freq_rel), 1),
            " %"
          ),
          paste0(
            "+/- 10 %: ",
            round(sum(Y_residuals_stat[10:11, ]$Freq_rel), 1),
            " %"
          )
        ),
        cex = legend.cex
      )
    }

    fit_metrics <- list(
      SQR = SQR,
      SAR = SAR,
      SQT = SQT,
      R2 = R2,
      MSE = MSE,
      RMSE = RMSE,
      MAE = MAE,
      MAPE = MAPE
    )

    invisible(
      list(
        fit_metrics = fit_metrics,
        observed_expected = observed_expected
      )
    )
  }


binary_metrics <- function(
  observed,
  expected,
  no_information_rate = "negative"
) {
  if (length(observed) != length(expected)) {
    stop("Vectors 'observed' and 'expected' differ in length")
  }

  if (!all(observed %in% c(0, 1)) || !all(expected %in% c(0, 1))) {
    stop("Observed and/or expected values are not binary")
  }

  observed_expected <- data.frame(observed, expected)
  observed_expected$hit <- 0
  observed_expected[
    observed_expected$observed == observed_expected$expected,
  ]$hit <- 1

  tab <- table(observed, expected)

  expected_required <- c("0", "1")
  expected_NA <- setdiff(expected_required, colnames(tab))

  if (length(expected_NA) > 0) {
    warning(paste0("No expected category for ", expected_NA), "\n")

    for (col in expected_NA) {
      tab <- cbind(tab, rep(0, nrow(tab)))

      colnames(tab)[ncol(tab)] <- col
    }

    tab <- tab[, expected_required]
  }

  sens <- tab[2, 2] / sum(tab[2, ])
  spec <- tab[1, 1] / sum(tab[1, ])
  acc <- sum(diag(tab)) / sum(tab)

  if (no_information_rate == "positive") {
    nir <- length(observed[observed == 1]) / length(observed)
  } else {
    nir <- length(observed[observed == 0]) / length(observed)
  }

  fit_metrics <-
    list(
      sens = sens,
      spec = spec,
      acc = acc,
      nir = nir
    )

  invisible(
    list(
      fit_metrics = fit_metrics,
      observed_expected = observed_expected
    )
  )
}


binary_metrics_glm <- function(
  logit_model,
  threshold = 0.5
) {
  y_pred <-
    ifelse(
      predict(
        logit_model,
        type = "response"
      ) >
        threshold,
      1,
      0
    )

  logit_metrics <- binary_metrics(
    observed = logit_model$y,
    expected = y_pred
  )

  invisible(logit_metrics)
}


breaks_growth <-
  function(
    y,
    t,
    ln = FALSE,
    add_constant = 1,
    alpha = 0.05,
    ...,
    verbose = FALSE
  ) {
    if (!is.numeric(y)) {
      stop("Infections vector must be of class 'numeric'.")
    }

    if (!is.numeric(t) & !is.Date(t)) {
      stop("Time vector must be of class 'numeric' or 'Date'.")
    }

    if (length(y) != length(t)) {
      stop(paste0(
        "Vectors 'y' and 't' differ in length: ",
        length(y),
        ", ",
        length(t)
      ))
    }

    yt_df <- data.frame(y, t)
    yt_df_length_clean <- yt_df[complete.cases(yt_df), ]

    if (nrow(yt_df_length_clean) < nrow(yt_df)) {
      y <- yt_df$y
      t <- yt_df$t

      nan_cases <- nrow(yt_df) - nrow(yt_df_length_clean)
      warning(paste0(nan_cases, " NaN values were removed"))
    }

    if (any(y <= 0)) {
      if (!is.null(add_constant) && is.numeric(add_constant)) {
        y_lin <- y + add_constant

        message(paste0(
          "Input infections data contains values <= 0. All values are increased by a constant equal to ",
          add_constant,
          "."
        ))
      } else {
        y_lin <- y[y > 0]

        warning(paste0(
          "Input infections data contains values <= 0. These values are skipped."
        ))
      }
    } else {
      y_lin <- y
    }

    TP <- length(y_lin)

    class_t <- "numeric"

    if (is.Date(t)) {
      class_t <- "Date"

      message("Time vector is of class 'Date'. Calculating time counter.")

      start_date <- min(t)

      time_counter <- as.integer(t - start_date)

      t <- time_counter
    }

    bp_formula <- as.formula("y ~ t")
    if (isTRUE(ln)) {
      bp_formula <- as.formula("log(y) ~ t")
    }

    if (isTRUE(verbose)) {
      if (isTRUE(ln)) {
        cat(paste0(
          "Calculating breakpoints for time series with ",
          TP,
          " time points, nat. log. of y values ... "
        ))
      } else {
        cat(paste0(
          "Calculating breakpoints for time series with ",
          TP,
          " time points ... "
        ))
      }
    }

    bp_obj <-
      breakpoints(
        formula = bp_formula,
        data = yt_df_length_clean,
        ...
      )

    bp_obj_coef <- coef(bp_obj)
    bp_obj_breakpoints <- bp_obj$breakpoints
    bp_obj_breakpoints_no <- length(bp_obj_breakpoints)
    bp_obj_segments_no <- bp_obj_breakpoints_no + 1

    if (bp_obj_breakpoints_no > 0) {
      bp_obj_breakpoints_ci <- confint(
        bp_obj,
        level = 1 - alpha
      )
    }

    if (isTRUE(verbose)) {
      cat("OK", "\n")
      cat(
        paste0(
          bp_obj_breakpoints_no,
          " breakpoints were found, which leads to ",
          bp_obj_segments_no,
          " segments"
        ),
        "\n"
      )
      cat(paste0("Calculating segment models ... "))
    }

    bp_obj_segments <-
      matrix(
        ncol = 2,
        nrow = bp_obj_segments_no
      )

    bp_obj_segments[1, ] <- c(1, bp_obj_breakpoints[1])

    i <- 0

    for (i in 2:(bp_obj_segments_no - 1)) {
      bp_obj_segments[i, ] <-
        c(
          (bp_obj_breakpoints[i - 1] + 1),
          bp_obj_breakpoints[i]
        )
    }

    bp_obj_segments[bp_obj_segments_no, ] <-
      c(
        bp_obj_breakpoints[bp_obj_breakpoints_no],
        length(y)
      )

    i <- 0

    slopes <- matrix(ncol = 3, nrow = bp_obj_segments_no)
    slopes_p <- vector()
    intercepts <- matrix(ncol = 3, nrow = bp_obj_segments_no)
    intercepts_p <- vector()
    SQR <- vector()
    SAR <- vector()
    SQT <- vector()
    R2 <- vector()
    MSE <- vector()
    RMSE <- vector()
    MAE <- vector()
    MAPE <- vector()
    segment_model_ci_points <- list()

    for (i in 1:bp_obj_segments_no) {
      yt_df_length_clean_segment <- yt_df_length_clean[
        bp_obj_segments[i, 1]:bp_obj_segments[i, 2],
      ]

      segment_model <-
        lm(
          bp_formula,
          data = yt_df_length_clean_segment
        )

      slopes[i, 1] <- segment_model$coefficients[2]
      intercepts[i, 1] <- segment_model$coefficients[1]

      segment_model_ci_est <-
        confint(
          segment_model,
          level = 1 - alpha
        )
      slopes[i, 2] <- segment_model_ci_est[2]
      slopes[i, 3] <- segment_model_ci_est[4]
      intercepts[i, 2] <- segment_model_ci_est[1]
      intercepts[i, 3] <- segment_model_ci_est[3]

      segment_model_summary <- summary(segment_model)

      intercepts_p[i] <- segment_model_summary$coefficients[7]
      slopes_p[i] <- segment_model_summary$coefficients[8]

      segment_model_y <- yt_df_length_clean_segment$y
      segment_model_y_pred <- predict(segment_model)
      if (isTRUE(ln)) {
        segment_model_y_pred <- exp(segment_model_y_pred)
      }

      segment_model_fit_metrics <- metrics(
        segment_model_y,
        segment_model_y_pred,
        plot = FALSE
      )
      segment_model_fit_metrics <- segment_model_fit_metrics[[1]]
      SQR[i] <- segment_model_fit_metrics$SQR
      SAR[i] <- segment_model_fit_metrics$SAR
      SQT[i] <- segment_model_fit_metrics$SQT
      R2[i] <- segment_model_fit_metrics$R2
      MSE[i] <- segment_model_fit_metrics$MSE
      RMSE[i] <- segment_model_fit_metrics$RMSE
      MAE[i] <- segment_model_fit_metrics$MAE
      MAPE[i] <- segment_model_fit_metrics$MAPE
      segment_model_fit_metrics <- NULL

      segment_model_ci_points[[i]] <-
        predict(
          segment_model,
          interval = "confidence",
          level = 1 - alpha
        )
    }

    segments_models_df <-
      cbind(
        bp_obj_segments,
        intercepts,
        intercepts_p,
        slopes,
        slopes_p,
        SQR,
        SAR,
        SQT,
        R2,
        MSE,
        RMSE,
        MAE,
        MAPE
      )

    colnames(segments_models_df) <-
      c(
        "first",
        "last",
        "intercept",
        "intercept_CI25",
        "intercept_CI975",
        "intercept_p",
        "slope",
        "slope_CI25",
        "slope_CI975",
        "slope_p",
        "SQR",
        "SAR",
        "SQT",
        "R2",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE"
      )

    model_breakpoints_list <-
      list(
        segments_models_df = segments_models_df,
        segment_model_ci_points = segment_model_ci_points,
        bp_obj_coef = bp_obj_coef,
        bp_obj_breakpoints = bp_obj_breakpoints,
        bp_obj_breakpoints_no = bp_obj_breakpoints_no,
        bp_obj_segments_no = bp_obj_segments_no,
        bp_obj_breakpoints_ci = bp_obj_breakpoints_ci,
        bp_obj = bp_obj
      )

    config <-
      list(
        TP = TP,
        ln = ln,
        add_constant = add_constant,
        alpha = alpha
      )

    if (isTRUE(verbose)) {
      cat("OK", "\n")
    }

    new(
      "breaksgrowth",
      GrowthModel_OLS = model_breakpoints_list,
      y = y,
      t = t,
      config = config
    )
  }


# Class growthmodels:
setClass(
  "growthmodels",
  slots = list(
    results = "data.frame",
    growth_models = "list",
    model_type = "character",
    results_cols = "character",
    results_cols_names = "character",
    data_statistics = "numeric",
    time_format = "character",
    timestamp = "list"
  ),
  prototype = list(
    timestamp = list()
  )
)

# Methods for class growthmodels:
setMethod(
  "summary",
  "growthmodels",
  function(object) {
    model_type <- object@model_type
    time_format <- object@time_format
    results_cols <- object@results_cols
    results_cols_names <- object@results_cols_names

    cat(paste0(model_type, " Growth Model"), "\n\n")

    cat("Input data\n")
    cat(sprintf("  Units       : %s\n", object@data_statistics[1]))
    cat(sprintf("  Time points : %s\n", object@data_statistics[2]))
    cat(sprintf(
      "  Balanced    : %s\n\n",
      ifelse(object@data_statistics[4], "YES", "NO")
    ))

    results_show <- object@results[results_cols]

    results_show[, 2:ncol(results_show)] <- round(
      results_show[, 2:ncol(results_show)],
      3
    )
    colnames(results_show) <- results_cols_names

    cat("Results per region")

    if (nrow(results_show) > 10) {
      cat(
        paste0(
          " (Showing first and last 5 of ",
          object@data_statistics[1],
          " cases)"
        ),
        "\n\n"
      )

      print(head(results_show, 5), row.names = FALSE)
      cat("...", "\n")
      print(tail(results_show, 5), row.names = FALSE)

      cat("\n")
      cat("Use your_growth_models@results to access the full table", "\n")
    } else {
      cat("\n\n")
      print(results_show[results_cols], row.names = FALSE)
    }

    invisible(object)
  }
)

setMethod(
  "print",
  "growthmodels",
  function(x) {
    cat(
      paste0(
        x@model_type,
        " Growth Models for ",
        x@data_statistics[1],
        " spatial units and ",
        x@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")

    invisible(x)
  }
)

setMethod(
  "show",
  "growthmodels",
  function(object) {
    cat(
      paste0(
        object@model_type,
        " Growth Models for ",
        object@data_statistics[1],
        " spatial units and ",
        object@data_statistics[2],
        " time points"
      ),
      "\n"
    )
    cat("Use summary() for results", "\n")

    invisible(object)
  }
)

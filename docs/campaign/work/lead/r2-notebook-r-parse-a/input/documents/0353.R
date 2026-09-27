daisy_manhattan_method <- function(dt, clusters, columnClass, metric) {
  start.time <- Sys.time()

  if (inherits(dt, 'data.frame')) {
    dt <- as.matrix(dt)
  }

  numeric_cluster <- ifelse(!is.numeric(clusters), 1, 0)

  if (sum(numeric_cluster) > 0) {
    stop('The field clusters must be a numeric')
  }

  daisy_manhattan <- tryCatch(
    {
      daisy(x = dt, metric = CONST_MANHATTAN)
    },

    error = function(cond) {
      return(CONST_NULL)
    }
  )

  if (!is.null(daisy_manhattan)) {
    daisy_manhattan_clust <-
      tryCatch(
        {
          hclust(dist(daisy_manhattan), method = CONST_SINGLE)
        },

        error = function(daisy_manhattan_clust) {
          return(CONST_NULL)
        }
      )

    if (!is.null(daisy_manhattan_clust)) {
      ev_daisy_manhattan <-
        tryCatch(
          {
            external_validation(
              c(dt[, columnClass]),
              cutree(daisy_manhattan_clust, k = clusters),
              metric
            )
          },

          error = function(daisy_manhattan_clust) {
            ev_daisy_manhattan <- initializeExternalValidation()
          }
        )

      iv_daisy_manhattan <- tryCatch(
        {
          internal_validation(
            distance = CONST_NULL,
            clusters_vector = cutree(daisy_manhattan_clust, k = clusters),
            dataf = dt,
            method = CONST_MANHATTAN,
            metric
          )
        },

        error = function(daisy_manhattan_clust) {
          iv_daisy_manhattan <- initializeInternalValidation()
        }
      )
    } else {
      ev_daisy_manhattan <- initializeExternalValidation()
      iv_daisy_manhattan <- initializeInternalValidation()
    }
  } else {
    ev_daisy_manhattan <- initializeExternalValidation()
    iv_daisy_manhattan <- initializeInternalValidation()
  }

  end.time <- Sys.time()
  time <- end.time - start.time

  ev_daisy_manhattan$time <- time - iv_daisy_manhattan$time
  iv_daisy_manhattan$time <- time - ev_daisy_manhattan$time

  result <- list(
    "external" = ev_daisy_manhattan,
    "internal" = iv_daisy_manhattan
  )

  return(result)
}
daisy_euclidean_method <- function(dt, clusters, columnClass, metric) {
  start.time <- Sys.time()

  if (inherits(dt, 'data.frame')) {
    dt <- as.matrix(dt)
  }

  numeric_cluster <- ifelse(!is.numeric(clusters), 1, 0)

  if (sum(numeric_cluster) > 0) {
    stop('The field clusters must be a numeric')
  }

  daisy_euclidean <- tryCatch(
    {
      daisy(x = dt, metric = CONST_EUCLIDEAN)
    },

    error = function(cond) {
      return(CONST_NULL)
    }
  )

  if (!is.null(daisy_euclidean)) {
    daisy_euclidean_clust <- hclust(
      dist(daisy_euclidean),
      method = CONST_CENTROID
    )

    if (!is.null(daisy_euclidean_clust)) {
      ev_daisy_euclidean <-
        tryCatch(
          {
            external_validation(
              c(dt[, columnClass]),
              cutree(daisy_euclidean_clust, k = clusters),
              metric
            )
          },

          error = function(cond) {
            ev_daisy_euclidean <- initializeExternalValidation()
          }
        )

      iv_daisy_euclidean <- tryCatch(
        {
          internal_validation(
            distance = CONST_NULL,
            clusters_vector = cutree(daisy_euclidean_clust, k = clusters),
            dataf = dt,
            method = CONST_EUCLIDEAN,
            metric
          )
        },

        error = function(cond) {
          iv_daisy_euclidean <- initializeInternalValidation()
        }
      )
    } else {
      ev_daisy_euclidean <- initializeExternalValidation()
      iv_daisy_euclidean <- initializeInternalValidation()
    }
  } else {
    ev_daisy_euclidean <- initializeExternalValidation()
    iv_daisy_euclidean <- initializeInternalValidation()
  }

  end.time <- Sys.time()
  time <- end.time - start.time

  ev_daisy_euclidean$time <- time - iv_daisy_euclidean$time
  iv_daisy_euclidean$time <- time - ev_daisy_euclidean$time

  result <- list(
    "external" = ev_daisy_euclidean,
    "internal" = iv_daisy_euclidean
  )

  return(result)
}
daisy_gower_method <- function(dt, clusters, columnClass, metric) {
  start.time <- Sys.time()

  if (inherits(dt, 'data.frame')) {
    dt <- as.matrix(dt)
  }

  numeric_cluster <- ifelse(!is.numeric(clusters), 1, 0)

  if (sum(numeric_cluster) > 0) {
    stop('The field clusters must be a numeric')
  }

  daisy_gower <- tryCatch(
    {
      daisy(x = dt, metric = CONST_GOWER)
    },

    error = function(cond) {
      return(CONST_NULL)
    }
  )

  if (!is.null(daisy_gower)) {
    daisy_gower_clust <-
      tryCatch(
        {
          hclust(dist(daisy_gower), method = CONST_SINGLE)
        },

        error = function(cond) {
          return(CONST_NULL)
        }
      )

    if (!is.null(daisy_gower_clust)) {
      ev_daisy_gower <-
        tryCatch(
          {
            external_validation(
              c(dt[, columnClass]),
              cutree(daisy_gower_clust, k = clusters),
              metric
            )
          },

          error = function(cond) {
            ev_daisy_gower <- initializeExternalValidation()
          }
        )

      iv_daisy_gower <- tryCatch(
        {
          internal_validation(
            distance = as.matrix(daisy_gower),
            clusters_vector = cutree(daisy_gower_clust, k = clusters),
            dataf = dt,
            method = CONST_NULL,
            metric
          )
        },

        error = function(cond) {
          iv_daisy_gower <- initializeInternalValidation()
        }
      )
    } else {
      ev_daisy_gower <- initializeExternalValidation()
      iv_daisy_gower <- initializeInternalValidation()
    }
  } else {
    ev_daisy_gower <- initializeExternalValidation()
    iv_daisy_gower <- initializeInternalValidation()
  }

  end.time <- Sys.time()
  time <- end.time - start.time

  ev_daisy_gower$time <- time - iv_daisy_gower$time
  iv_daisy_gower$time <- time - ev_daisy_gower$time

  result <- list("external" = ev_daisy_gower, "internal" = iv_daisy_gower)

  return(result)
}
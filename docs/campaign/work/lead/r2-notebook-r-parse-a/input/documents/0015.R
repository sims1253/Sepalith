uwt <- function(
  svysmpl,
  svyVar,
  svypopu = NULL,
  subset = NULL,
  family = gaussian(),
  invlvls,
  weights = NULL
) {
  subset <- c("T", subset)
  if (is.null(svypopu)) {
    if (is.null(weights)) {
      des <- survey::svydesign(ids = ~1, weights = ~1, data = svysmpl)
    } else {
      des <- survey::svydesign(ids = ~1, weights = ~weights, data = svysmpl)
    }
  } else {
    message(
      "population parameter is specified, so the finite population correction will be calculated for sample mean.\n"
    )
    svysmpl$fpc <- nrow(svypopu)
    if (is.null(weights)) {
      des <- survey::svydesign(
        ids = ~1,
        weights = ~1,
        data = svysmpl,
        fpc = ~fpc
      )
    } else {
      des <- survey::svydesign(
        ids = ~1,
        weights = ~weights,
        data = svysmpl,
        fpc = ~fpc
      )
    }
  }
  infr <- sapply(
    subset,
    function(s) {
      des <- subset(des, eval(parse(text = s)))

      if (family$family == "binomial") {
        suppressWarnings(
          desc <- sapply(
            invlvls,
            function(lv) {
              paste0('~', svyVar) %>%
                as.formula() %>%
                survey::svyciprop(des, method = "logit", level = lv)
            },
            simplify = FALSE
          )
        )
        tCI <- lapply(desc, function(i) {
          ci <- confint(i, df = survey::degf(des), parm = svyVar)
          colnames(ci) <- stringr::str_replace(colnames(ci), "%", " %")
          ci
        })

        tCI <- do.call("cbind", tCI)
        desc <- desc[[1]]
      }
      if (family$family == "gaussian") {
        desc <- paste0('~', svyVar) %>% as.formula() %>% survey::svymean(des)
        tCI <- sapply(
          invlvls,
          confint,
          object = desc,
          parm = svyVar,
          simplify = FALSE
        )
        tCI <- do.call("cbind", tCI)
      }

      # get estmates and standard error
      if (!is.null(svypopu)) {
        infr <- cbind(
          est = desc[svyVar],
          se = sqrt(diag(vcov(desc))),
          tCI,
          sample_size = survey::degf(des) + 1,
          population_size = nrow(dplyr::filter(svypopu, eval(parse(text = s))))
        )
      } else {
        infr <- cbind(
          est = desc[svyVar],
          se = sqrt(diag(vcov(desc))),
          tCI,
          sample_size = survey::degf(des) + 1
        )
      }
      if (is.null(weights)) {
        rownames(infr) <- "sample_mean"
      } else {
        rownames(infr) <- "Weighted-sample_mean"
      }
      infr
    },
    simplify = FALSE
  )
  names(infr)[1] <- "All"
  if (length(infr) == 1) {
    return(infr[[1]])
  }
  return(infr)
}
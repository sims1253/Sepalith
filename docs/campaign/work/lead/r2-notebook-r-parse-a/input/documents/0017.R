postStr_wt <- function(
  svysmpl,
  svypopu,
  auxVars,
  svyVar,
  subset = NULL,
  family = gaussian(),
  invlvls,
  weights = NULL
) {
  subset <- c("T", subset)
  svysmpl$fpc <- nrow(svypopu) # get population sample size

  # specify svydesign
  if (is.null(weights)) {
    des <- survey::svydesign(ids = ~1, weights = ~1, data = svysmpl, fpc = ~fpc)
  } else {
    des <- survey::svydesign(
      ids = ~1,
      weights = ~weights,
      data = svysmpl,
      fpc = ~fpc
    )
  }

  fmla <- paste(auxVars, collapse = "+")
  fmla <- paste('~', fmla, sep = ' ') %>% as.formula()

  # set for post-stratification
  tab <- xtabs(fmla, svypopu)
  PSobj <- survey::postStratify(des, fmla, tab, partial = TRUE)
  #d = population[population$Z3 == 0 &population$auX_5  == 1,  ]
  #table(d$Z1, d$Z2)
  infr <- sapply(
    subset,
    function(s) {
      # make auxVar as a readable formula, for example, if auxVar = c(Z1, Z2, Z3), the result formula is ~Z1 + Z2 + Z3
      PSobj <- subset(PSobj, eval(parse(text = s)))
      #svytable(fmla, PSobj, round = TRUE)

      # get estimates and confidence intervals
      # this function allows users specify multiple confidence levels, such as invlvls = c(0.95, 0.8)
      # so sapply function will calculate every confidence intervals separately
      if (family$family == "binomial") {
        suppressWarnings(
          PSest <- sapply(
            invlvls,
            function(lv) {
              paste0('~', svyVar) %>%
                as.formula() %>%
                survey::svyciprop(PSobj, method = "logit", level = lv)
            },
            simplify = FALSE
          )
        )
        tCI <- lapply(PSest, function(i) {
          ci <- confint(i, df = survey::degf(PSobj), parm = svyVar)
          colnames(ci) <- str_replace(colnames(ci), "%", " %")
          ci
        })
        tCI <- do.call("cbind", tCI)
        PSest <- PSest[[1]]
      }
      if (family$family == "gaussian") {
        PSest <- paste0('~', svyVar) %>% as.formula() %>% survey::svymean(PSobj)
        tCI <- sapply(
          invlvls,
          confint,
          object = PSest,
          df = survey::degf(PSobj),
          parm = svyVar,
          simplify = FALSE
        )
        tCI <- do.call("cbind", tCI)
      }
      # get estmates and standard error
      infr <- cbind(
        est = PSest[svyVar],
        se = sqrt(diag(vcov(PSest))),
        tCI,
        sample_size = survey::degf(PSobj) + 1,
        population_size = nrow(dplyr::filter(svypopu, eval(parse(text = s))))
      )
      if (is.null(weights)) {
        rownames(infr) <- "postStratify"
      } else {
        rownames(infr) <- "Weighted-postStratify"
      }
      return(infr)
    },
    simplify = FALSE
  )
  names(infr)[1] <- "All"
  if (length(infr) == 1) {
    return(infr[[1]])
  }
  return(infr)
}
rake_wt <- function(
  svysmpl,
  svypopu,
  auxVars,
  svyVar,
  subset = NULL,
  family = gaussian(),
  invlvls,
  weights = NULL,
  maxiter = 50
) {
  subset <- c("T", subset)
  svysmpl$fpc <- nrow(svypopu)
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
  popu_tab <- lapply(auxVars, function(Var) {
    Var <- paste(all.vars(as.formula(paste0("~", Var))), collapse = "+")
    fmla <- paste('~', Var, sep = ' ') %>% as.formula()
    tab <- xtabs(fmla, svypopu)
    return(tab)
  })
  fmla_list <- lapply(auxVars, function(Var) {
    Var <- paste(all.vars(as.formula(paste0("~", Var))), collapse = "+")
    paste('~', Var, sep = ' ') %>% as.formula() %>% return()
  })
  rakingobj <- survey::rake(
    des,
    fmla_list,
    popu_tab,
    control = list(maxit = maxiter, epsilon = 1, verbose = FALSE)
  )
  infr <- sapply(
    subset,
    function(s) {
      rakingobj <- subset(rakingobj, eval(parse(text = s)))
      if (family$family == "binomial") {
        suppressWarnings(
          rakest <- sapply(
            invlvls,
            function(lv) {
              paste0('~', svyVar) %>%
                as.formula() %>%
                survey::svyciprop(rakingobj, method = "logit", level = lv)
            },
            simplify = FALSE
          )
        )
        tCI <- lapply(rakest, function(i) {
          ci <- confint(i, df = survey::degf(rakingobj), parm = svyVar)
          colnames(ci) <- stringr::str_replace(colnames(ci), "%", " %")
          ci
        })
        tCI <- do.call("cbind", tCI)
        rakest <- rakest[[1]]
      }
      if (family$family == "gaussian") {
        rakest <- paste0('~', svyVar) %>%
          as.formula() %>%
          survey::svymean(rakingobj)
        tCI <- sapply(
          invlvls,
          confint,
          object = rakest,
          df = survey::degf(rakingobj),
          parm = svyVar,
          simplify = FALSE
        )
        tCI <- do.call("cbind", tCI)
      }
      infr <- cbind(
        est = rakest[svyVar],
        se = sqrt(diag(vcov(rakest))),
        tCI,
        sample_size = survey::degf(rakingobj) + 1,
        population_size = nrow(dplyr::filter(svypopu, eval(parse(text = s))))
      )
      if (is.null(weights)) {
        rownames(infr) <- "rake"
      } else {
        rownames(infr) <- "Weighted-rake"
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
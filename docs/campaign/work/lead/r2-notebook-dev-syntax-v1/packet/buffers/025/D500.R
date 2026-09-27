globalVariables(c("variable"))

extract.knots <- function(res.glm) {
  if (any(class(res.glm) %in% c("biglm", "bigglm"))) {
    tmp <- dimnames(attr(res.glm$terms, "factors"))[[2]]
    name.v <- as.character(tmp)
    names(name.v) <- stringr::str_extract(
      name.v,
      "(?:[a-zA-Z]+\\()?([a-zA-Z][a-zA-Z0-9_\\.]+)",
      group = 1
    )
    res <- lapply(rlang::set_names(seq_along(name.v), names(name.v)), \(x) {
      res0 <- list(v = names(name.v)[x])
      if (stringr::str_detect(tmp[[x]], "knots")) {
        kn <- paste0(
          "knots(",
          stringr::str_replace(tmp[[x]], names(name.v)[x], "0"),
          ")"
        ) |>
          parse(text = _) |>
          eval()
        res0 <- c(
          res0,
          list(
            knots = kn[-c(1, length(kn))],
            Boundary.knots = kn[c(1, length(kn))]
          )
        )
      }
    })
  } else {
    tmp <- attr(res.glm$terms, "predvars")
    name.v <- as.character(tmp)
    names(name.v) <- stringr::str_extract(
      name.v,
      "(?:[a-zA-Z]+\\()?([a-zA-Z][a-zA-Z0-9_\\.]+)",
      group = 1
    )
    res <- lapply(rlang::set_names(seq_along(name.v), names(name.v)), \(x) {
      res0 <- list(v = names(name.v)[x])
      if (any("knots" %in% names(tmp[[x]]))) {
        res0 <- c(
          res0,
          list(knots = tmp[[x]]$knots, Boundary.knots = tmp[[x]]$Boundary.knots)
        )
      }
    })
  }
  res[-which(name.v == "list" | is.na(names(name.v)) | sapply(res, is.null))]
}

#' Title Pool meta-analysis estimates and estimates from a regression model.
#'
#' @param v Name of the covariate, which is modeled using an `nsk` spline (see package `splines2`).
#' @param meta.df Meta-analysis estimates: dataframe with columns `est` (e.g. log HR estimate), `est.var` (estimated variance), `variable` (name of the covariate used in the spline) and `cov.value` (covariate value at which est and est.var were reported).
#' @param glm.res Regression analysis result object.
#' @param cor.m Assumed correlation matrix. If NULL (default) or NA then use correlation matrix from `glm.res`.
#' @param x.range If NULL (default), then take the range from `meta.df`, otherwise give range as a vector with two components.
#' @param full.output If TRUE then output also the log HR values and 95% confidence intervals over a grid of covariate values.
#'
#' @return List containing pooled estimates of the spline parameters.
#' @export
pool_splines <- function(
  v,
  meta.df,
  glm.res,
  cor.m = NULL,
  x.range = NULL,
  full.output = FALSE
) {
  idx.param <- which(
    stringr::str_detect(
      names(stats::coef(glm.res)),
      paste0("[^a-zA-Z0-9_\\.]?", v, "[^a-zA-Z0-9_\\.]?")
    ) &
      stringr::str_detect(names(stats::coef(glm.res)), "^nsk\\(")
  )
  if (length(idx.param) > 0 && any(meta.df$variable == v)) {
    icept <- FALSE
    # for splines:
    knot.l <- extract.knots(glm.res)[[v]]
    idx.ref <- which(meta.df$cov.value == knot.l$Boundary.knots[1])[1] |>
      tidyr::replace_na(1)
    nsk.base <- stringr::str_replace(
      names(stats::coef(glm.res))[idx.param[1]],
      v,
      "meta.df$cov.value"
    ) |>
      stringr::str_replace("[0-9]+$", "") |>
      parse(text = _) |>
      eval()
    # sweep to make contrast w.r.t. to left boundary knot:
    nsk.base <- sweep(nsk.base, 2, nsk.base[idx.ref, ])
    x <- seq(min(meta.df$cov.value), max(meta.df$cov.value), length = 50)
    # least squares as the loss function optimization for spline parameters:
    # (i.e. inverse transformation from the g/d scale to the spline parameters)
    f <- function(x) {
      idx <- !is.na(meta.df$est)
      sum((meta.df$est[idx] - nsk.base[idx, ] %*% x)^2) # / ((meta.df$est.var)+.01))
    }
    # est.beta <- stats::optim(rep(0,ncol(nsk.base)), f)
    est.beta <- optimization::optim_nm(
      f,
      k = ncol(nsk.base),
      start = rep(0, ncol(nsk.base))
    )
    # cat("est:", est.beta$par, "\n")
    # find standard errors:
    if (is.null(cor.m) || is.na(cor.m)) {
      # get correlation matrix from the glm result:
      cor.m2 <- suppressWarnings(stats::cov2cor(stats::vcov(glm.res)))[
        idx.param,
        idx.param
      ]
    } else {
      if (is.character(cor.m)) {
        # evaluate from string:
        cor.m2 <- eval(parse(text = cor.m))
      }
      if (is.matrix(cor.m)) {
        # matrix:
        cor.m2 <- cor.m
      }
    }
    f2 <- function(x) {
      # covariance matrix of spline parameters in terms of standard errors:
      var.m2 <- outer(x, x) * cor.m2
      # covariance matrix of the spline values at reported servings:
      var.spline.values <- (nsk.base) %*% var.m2 %*% t(nsk.base)
      # least squares of the diagonal values (spline component variances) as the loss function:
      # sum((meta.df$est.var - diag(var.spline.values))^2) # minimize w.r.t. variance
      sum(((meta.df$est.var)^(1 / 4) - (diag(var.spline.values)^(1 / 4)))^2) # minimize w.r.t. root of standard error
    }
    est.beta2 <- stats::optim(
      diag(stats::vcov(glm.res)[idx.param, idx.param]),
      # rep(0.01, ncol(nsk.base)),
      f2,
      method = "L-BFGS-B",
      # set lower limit close to zero:
      lower = min(diag(stats::vcov(glm.res)[idx.param, idx.param])) / 1e6
    )
    # cat("se:", est.beta2$par, "\n")

    tmp.var.m <- outer((est.beta2$par), (est.beta2$par)) * cor.m2
    dimnames(tmp.var.m) <- NULL

    # single-parameter meta-analyses:
    # cat("est:", est.beta$par, "\n"); cat("var:", est.beta2$par^2, "\n")
    res <- list(
      pooled = sapply(seq_along(idx.param), \(i) {
        tmp.est <- c(
          lb = unname(est.beta$par[i]),
          ipd = unname(stats::coef(glm.res)[idx.param[i]])
        )
        tmp.var <- c(
          lb = unname(est.beta2$par[i]^2),
          ipd = unname(diag(stats::vcov(glm.res))[idx.param[i]])
        )
        if (unname(est.beta2$par[i]^2) > 1e-8) {
          # clearly positive variance - make meta-analysis:
          res0 <- meta::metagen(tmp.est, sqrt(tmp.var))
        } else {
          # optim returned almost zero variance - use meta-analysis estimates:
          res0 <- list(
            TE.common = unname(est.beta$par[i]),
            seTE.common = sqrt(unname(est.beta2$par[i]^2)),
            pval.Q = NA
          )
        }
        c(
          est = tmp.est,
          var = tmp.var,
          est.pool = res0$TE.common,
          var.pool = res0$seTE.common^2,
          pval.Q = res0$pval.Q
        )
      }) |>
        as.data.frame() |>
        tibble::rownames_to_column("stat")
    )
    attr(res, "idx.param") <- idx.param

    if (full.output) {
      tmp.f <- function(est, c.m, m) {
        est.v <- m %*% est
        se.v <- sqrt(diag(m %*% c.m %*% t(m)))
        data.frame(
          est = est.v,
          ci.low = est.v - 1.96 * se.v,
          ci.upp = est.v + 1.96 * se.v
        )
      }
      x2 <- x
      if (!is.null(x.range)) {
        x2 <- seq(x.range[1], x.range[2], length = 50)
      }
      # design matrix:
      nsk.base2 <- stringr::str_replace(
        names(stats::coef(glm.res))[idx.param[1]],
        v,
        "x2"
      ) |>
        stringr::str_replace("[0-9]+$", "") |>
        parse(text = _) |>
        eval()
      # contrast w.r.t. first row:
      nsk.base2 <- sweep(nsk.base2, 2, nsk.base2[1, ])
      log.hr <- dplyr::bind_rows(
        cbind(
          model = "ipd",
          cov.value = x2,
          tmp.f(
            stats::coef(glm.res)[idx.param],
            stats::vcov(glm.res)[idx.param, idx.param],
            nsk.base2
          )
        ),
        cbind(
          model = "lb",
          cov.value = x2,
          tmp.f(est.beta$par, tmp.var.m, nsk.base2)
        ),
        cbind(
          model = "pooled",
          cov.value = x2,
          tmp.f(
            unlist(res$pooled[res$pooled$stat == "est.pool", -1]),
            (outer(
              unlist(sqrt(res$pooled[res$pooled$stat == "var.pool", -1])),
              unlist(sqrt(res$pooled[res$pooled$stat == "var.pool", -1]))
            ) *
              cor.m2),
            nsk.base2
          )
        )
      )
      res <- c(
        res,
        list(log.hr = log.hr, meta.df = meta.df, meta.vcov = tmp.var.m)
      )
    }
  } else {
    res <- NULL
  }
  res
}

#' Title Pool meta-analysis estimates and estimates from a regression model.
#'
#' @param v Name of the covariate, which is modeled using an nsk spline.
#' @param meta.df Meta-analysis estimates: dataframe with columns variable (covariate name), est (log HR estimate), est.var (estimated variance) and cov.value (covariate values where est and est.var were reported).
#' @param glm.res Regression analysis result object.
#'
#' @examples
#' # Estimate a linear regression model using an individual participant data (IPD):
#' library(metasplines)
#' library(splines2)
#' res <- lm(
#'   Petal.Width ~
#'     Species +
#'     nsk(Sepal.Length, Boundary.knots = c(4.5, 7.5), knots = c(5, 6, 6.5)),
#'   data=iris)
#' # "Literature-based" (LB) estimates:
#' lb.df <- read.table(text=
#' "variable,     cov.value,  est,  est.var
#' Sepal.Length,  4.5,	       0,     0
#' Sepal.Length,  5,	         0.15,  0.01
#' Sepal.Length,  5.5,	       0.25,  0.01
#' Sepal.Length,  6,	         0.4,   0.01
#' Sepal.Length,  6.5,	       0.5,   0.01
#' Sepal.Length,  8,          0.25,  0.04
#' ", sep=",", header=TRUE)
#' # Output table with the point estimates and the estimated variances:
#' pool_splines(v="Sepal.Length", meta.df=lb.df, glm.res=res)
#'
#' @return List containing pooled estimates of the spline parameters.
#' @export
pool_all_splines <- function(v, meta.df, glm.res) {
  res.l <- lapply(
    rlang::set_names(v),
    \(xx) {
      pool_splines(
        xx,
        meta.df = meta.df |> dplyr::filter(variable == xx),
        glm.res = glm.res,
        cor.m = NULL,
        x.range = NULL,
        full.output = FALSE
      )
    }
  )
  res.l <- res.l[!sapply(res.l, is.null)]
  # collect point and variance estimates:
  res <- lapply(names(res.l), \(x) {
    res.l[[x]]$pooled |>
      dplyr::mutate(variable = x) |>
      dplyr::relocate(variable)
  }) |>
    dplyr::bind_rows()
  # collect pooled point estimates and covariance matrix:
  est.v <- stats::coef(glm.res)
  var.v <- stats::vcov(glm.res) |> diag()
  cor.m <- suppressWarnings(stats::cov2cor(stats::vcov(glm.res)))
  for (i in names(res.l)) {
    est.v[attr(res.l[[i]], "idx.param")] <-
      as.matrix(res.l[[i]]$pooled[res.l[[i]]$pooled$stat == "est.pool", -1])
    var.v[attr(res.l[[i]], "idx.param")] <-
      as.matrix(res.l[[i]]$pooled[res.l[[i]]$pooled$stat == "var.pool", -1])
  }
  var.m <- outer(sqrt(var.v), sqrt(var.v)) * cor.m
  list(pooled = res, est = est.v, vcov = var.m)
}

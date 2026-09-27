node_inner.left_rotated_tree <- function(
  obj,
  id = TRUE,
  pval = TRUE,
  abbreviate = FALSE,
  fill = "white",
  gp = gpar()
) {
  meta <- obj$data
  nam <- names(obj)

  extract_label <- function(node) {
    if (is.terminal(node)) {
      return(rep.int("", 2L))
    }

    varlab <- character_split(split_node(node), meta)$name
    if (abbreviate > 0L) {
      varlab <- abbreviate(varlab, as.integer(abbreviate))
    }

    ## FIXME: make more flexible rather than special-casing p-value
    if (pval) {
      pval <- suppressWarnings(try(
        !is.null(info_node(node)$p.value),
        silent = TRUE
      ))
      pval <- if (inherits(pval, "try-error")) FALSE else pval
    }
    if (pval) {
      pvalue <- node$info$p.value
      plab <- ifelse(
        pvalue < 10^(-3L),
        paste("p <", 10^(-3L)),
        paste("p =", round(pvalue, digits = 3L))
      )
    } else {
      plab <- ""
    }
    return(c(varlab, plab))
  }

  maxstr <- function(node) {
    lab <- extract_label(node)
    klab <- if (is.terminal(node)) {
      ""
    } else {
      unlist(lapply(kids_node(node), maxstr))
    }
    lab <- c(lab, klab)
    lab <- unlist(lapply(lab, function(x) strsplit(x, "\n")))
    lab <- lab[which.max(nchar(lab))]
    if (length(lab) < 1L) {
      lab <- ""
    }
    return(lab)
  }

  nstr <- maxstr(node_party(obj))
  if (nchar(nstr) < 6) {
    nstr <- "aAAAAa"
  }

  ### panel function for the inner nodes
  rval <- function(node, ...) {
    # extract ...
    kwargs <- list(...)
    if (!is.null(kwargs$remove.nobs)) {
      remove.nobs <- kwargs$remove.nobs
    } else {
      remove.nobs <- FALSE
    }

    node_vp <- viewport(
      x = unit(0.5, "npc"),
      y = unit(0.5, "npc"),
      width = unit(1, "strwidth", nstr) + unit(0.2, "lines"),
      height = unit(3, "lines"),
      name = paste("node_inner.left_rotated_tree", id_node(node), sep = ""),
      gp = gp
    )
    pushViewport(node_vp)

    xell <- c(
      seq(0, 0.2, by = 0.01),
      seq(0.2, 0.8, by = 0.05),
      seq(0.8, 1, by = 0.01)
    )
    yell <- sqrt(xell * (1 - xell))

    lab <- extract_label(node)

    #fill <- rep(fill, length.out = 2L)
    # grid.polygon(x = unit(c(xell, rev(xell)), "npc"),#x = unit(0.1, "npc"), y = unit(0.1, "npc"),
    #              y = unit(c(yell, -yell)+0.5, "npc"),
    #              gp = gpar(fill = fill[1], col=fill[1]))

    grid.rect(height = unit(0.15, "lines"), gp = gpar(fill = fill, col = fill))

    ## FIXME: something more general instead of pval ?
    grid.text(lab[1L], y = unit(1.5 + 0.5 * (lab[2L] != ""), "lines")) #That's the variable name
    #grid.text('Test', y = unit(1, "lines")) #That's the variable name
    if (lab[2L] != "") {
      grid.text(lab[2L], y = unit(1, "lines"))
    } #Printing p-value

    if (id) {
      if (id_node(node) == 1) {
        nodeIDvp <- viewport(
          x = unit(0.5, "npc") + unit(0.5, "strwidth", nstr) + unit(0.1, "lines"),
          y = unit(0.5, "npc"),
          width = max(
            unit(1, "lines"),
            unit(1.3, "strwidth", nam[id_node(node)])
          ),
          height = max(
            unit(1, "lines"),
            unit(1.3, "strheight", nam[id_node(node)])
          ),
          just = "left"
        )
        pushViewport(nodeIDvp)
        grid.rect(gp = gpar(fill = fill))
        grid.text(nam[id_node(node)]) #print node number
        popViewport()

        # Print number of observations

        if (!remove.nobs) {
          dat <- data_party(obj, id_node(node))
          yn <- dat[["(response)"]]
          wn <- dat[["(weights)"]]
          if (is.null(wn)) {
            wn <- rep(1, NROW(yn))
          }
          nodeNum <- viewport(
            x = unit(0.5, "npc"),
            #- unit(.5, "strwidth", nstr) + unit(0.2, "lines")+ max(unit(1, "lines"), unit(1.3, "strwidth", nam[id_node(node)])),
            y = unit(1, "npc"),
            width = max(
              unit(0.8, "lines"),
              unit(1.3, "strwidth", sprintf("n = %s", sum(wn)))
            ),
            height = max(
              unit(1, "lines"),
              unit(1.3, "strheight", sprintf("n = %s", sum(wn)))
            )
          )
          pushViewport(nodeNum)
          # if (debug) {
          #   grid.rect(gp = gpar(fill = fill[2], lty="dotted", col="red"))
          # } else {
          grid.rect(gp = gpar(fill = fill[2], col = "white")) #
          # }
          grid.text(sprintf("n = %s", sum(wn)), just = c("center", "center")) #print node number
          popViewport()
        }
      }
    }
    upViewport()
  }

  return(rval)
}
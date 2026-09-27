mvregmed.graph.attributes <- function(fit.edges, x.color ="palegreen",
                                      y.color="palevioletred", 
                                      med.color="skyblue", v.size=30){
  fit.edges$color <- x.color
  fit.edges$label <- y.color
  fit.edges$med <- med.color
  fit.edges$size <- v.size
  return(fit.edges)
}
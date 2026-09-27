giant_component_extract <- function( x, directed = TRUE, bipartite_proj=FALSE ,num_proj=1){

  if (!("igraph" %in% class(x)|| "network" %in% class(x))) stop("The input is not an igraph or a network object")

  if (is_igraph(x)){

    if ( bipartite_proj){

      if (is_bipartite(x)){

        x <- bipartite.projection(x)[[num_proj]]

        if (!is_simple(x))   x <- simplify(x)

        cl <- clusters(x)

        giant_comp <- induced.subgraph(x, which(cl$membership == which.max(cl$csize)))

        giant_comp_edgelist <- as_edgelist(giant_comp, names = TRUE)

      }

      else stop("The graph is not bipartite")

    }

    else{

      if (!is_simple(x))   x<-simplify(x)

      cl <- clusters(x)

      giant_comp <- induced.subgraph(x, which(cl$membership == which.max(cl$csize)))

      giant_comp_edgelist <- as_edgelist(giant_comp, names = TRUE)

    }

  }

  if ( is.network(x)){

    edgelist<-as.edgelist(x)

    x <- graph_from_edgelist(edgelist, directed = directed)

    if (!is_simple(x))  x<-simplify(x)

    cl <- clusters(x)

    giant_comp <- induced.subgraph(x, which(cl$membership == which.max(cl$csize)))

    giant_comp_edgelist <- as_edgelist(giant_comp, names = TRUE)

  }

  result <- list(giant_comp,giant_comp_edgelist)

  return(result)

}
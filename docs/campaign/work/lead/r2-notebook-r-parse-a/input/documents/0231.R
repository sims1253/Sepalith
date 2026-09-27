search_string_db <- function(
  genes_list,
  species = 9606,
  network_type = "full",
  score_threshold = 400
) {
  if (length(genes_list) == 0) {
    warning("No genes provided for STRING database search.")
    return(list(
      string_results = data.frame(),
      string_db = NULL,
      string_ids = NULL
    ))
  }

  string_db <- STRINGdb$new(
    species = species,
    score_threshold = score_threshold,
    input_directory = "",
    network_type = network_type,
    version = "12.0"
  )

  mapped_genes <- string_db$map(data.frame(gene = genes_list), "gene")
  if (nrow(mapped_genes) == 0) {
    warning("No valid genes found in STRING database for provided genes_list.")
    return(list(
      string_results = data.frame(),
      string_db = string_db,
      string_ids = NULL
    ))
  }

  unique_mapped_genes <- mapped_genes %>% group_by(gene) %>% slice(1)
  string_ids <- unique_mapped_genes$STRING_id

  interactions <- string_db$get_interactions(string_ids)
  interaction_pairs <- data.table(
    proteinA = pmin(interactions$from, interactions$to),
    proteinB = pmax(interactions$from, interactions$to)
  )
  interaction_pairs <- unique(interaction_pairs)
  interaction_pairs <- interaction_pairs[
    interaction_pairs$proteinA != interaction_pairs$proteinB,
  ]

  if (nrow(interaction_pairs) == 0) {
    warning("No interactions found for the provided genes in STRING database.")
    return(list(
      string_results = data.frame(),
      string_db = string_db,
      string_ids = string_ids
    ))
  }

  nodes <- unique(c(
    interaction_pairs$proteinA,
    interaction_pairs$proteinB,
    string_ids
  ))
  adjacency_matrix <- matrix(
    0,
    nrow = length(nodes),
    ncol = length(nodes),
    dimnames = list(nodes, nodes)
  )

  for (i in seq_len(nrow(interaction_pairs))) {
    adjacency_matrix[
      interaction_pairs$proteinA[i],
      interaction_pairs$proteinB[i]
    ] <- 1
    adjacency_matrix[
      interaction_pairs$proteinB[i],
      interaction_pairs$proteinA[i]
    ] <- 1
  }

  degree <- rowSums(adjacency_matrix)
  clustering_coefficients <- numeric(length(nodes))
  for (i in seq_along(nodes)) {
    neighborhood <- which(adjacency_matrix[i, ] == 1)
    k <- length(neighborhood)
    if (k >= 2) {
      subgraph <- adjacency_matrix[neighborhood, neighborhood]
      triangles <- sum(subgraph) / 2
      clustering_coefficients[i] <- 2 * triangles / (k * (k - 1))
    }
  }

  graph_obj <- graph_from_adjacency_matrix(
    adjacency_matrix,
    mode = "undirected"
  )
  components <- components(graph_obj)

  gene_symbol_lookup <- setNames(
    unique_mapped_genes$gene,
    unique_mapped_genes$STRING_id
  )
  nodes_symbols <- sapply(nodes, function(node) {
    if (node %in% names(gene_symbol_lookup)) {
      gene_symbol_lookup[node]
    } else {
      # Return the STRING ID instead of NA
      node
    }
  })

  string_results <- data.frame(
    Gene_Symbol = nodes_symbols,
    Degree = degree,
    Clustering_Coefficient_Percent = clustering_coefficients * 100,
    Clustering_Coefficient_Fraction = sapply(seq_along(nodes), function(i) {
      k <- degree[i]
      if (k < 2) {
        return("0 / 0")
      } else {
        max_possible_edges <- k * (k - 1) / 2
        actual_edges <- clustering_coefficients[i] * max_possible_edges
        return(paste(round(actual_edges), "/", max_possible_edges, sep = ""))
      }
    }),
    Connected_Component_id = as.numeric(components$membership),
    Nodes_in_Connected_Component = as.numeric(components$csize[
      components$membership
    ]),
    total_number_of_connected_components = components$no,
    row.names = NULL # Add this line
  )

  string_results <- string_results %>%
    arrange(desc(Degree), desc(Clustering_Coefficient_Percent)) %>%
    mutate(Connectivity_Rank = row_number())

  return(list(
    string_results = string_results,
    string_db = string_db,
    string_ids = string_ids
  ))
}
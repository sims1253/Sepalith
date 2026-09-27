boundary_null_distrib <- function(x, calculate_intensity = FALSE, cat = FALSE, threshold = 0.2, n_iterations = 10, model = 'random', p = 0.5,
                                  progress = TRUE) {
  # progress bar
  if (progress == TRUE) {progress_bar = txtProgressBar(min = 0, max = n_iterations, initial = 0, char = '+', style = 3)}
  
  # if the model is Gaussian, estimate the autocorrelation range in the input raster
  if (model == 'gaussian') {
    pts <- terra::spatSample(x, size = length(terra::cells(x)), method = 'regular', na.rm = TRUE, as.points = TRUE) %>%
      terra::as.data.frame(geom = 'XY')
    colnames(pts)[1] <- 'value'
    vgm <- gstat::gstat(formula = value~1, locations = ~x+y, data = pts) %>%
      gstat::variogram()
    vgm <- suppressWarnings(gstat::fit.variogram(vgm, model = gstat::vgm(c('Exp', 'Sph', 'Gau', 'Mat'))))
    autocorr_range <- vgm$range[2]
  }

  # make output vectors for repeat loop
  n_boundary = c(); longest_boundary = c()

  # n iterations of random boundaries + stats
  rep = 0
  repeat {
    # simulate random raster with boundary elements ----
    repeat {
      if (model == 'random') {
        x_sim <- random_raster_sim(x)
      } else if (model == 'gaussian') {
        x_sim <- gauss_random_field_sim(x, autocorr_range)
      } else if (model == 'random_cluster') {
        x_sim <- mod_random_clust_sim(x, p)
      }

      x_boundary <- suppressMessages(define_boundary(x_sim, cat, threshold, calculate_intensity))

      if (sum(terra::values(x_boundary) == 1, na.rm = TRUE) >= 1) {break}
    }

    # number of subgraphs ----
    count <- terra::patches(x_boundary, directions = 8, zeroAsNA = TRUE) %>%
      terra::values() %>%
      max(na.rm = TRUE)

    n_boundary = append(n_boundary, count)

    # maximum length of a subgraph ----
    # organize boundary elements into separate boundaries based on which boundary elements are adjacent
    boundary_patches <- terra::patches(x_boundary, directions = 8, zeroAsNA = TRUE)
    patch_ids <- terra::values(boundary_patches) %>%
      unique %>%
      na.omit %>%
      as.vector
    
    # for each boundary/subgraph
    lengths <- c()
    for (i in patch_ids) {
      
      # find the farthest two cells in the boundary
      cell_coord <- terra::xyFromCell(boundary_patches, terra::cells(boundary_patches, i)$patches)
      rownames(cell_coord) <- terra::cells(boundary_patches, i)$patches
      dist.matrix <- as.matrix(terra::distance(cell_coord, lonlat = T))
      rownames(dist.matrix) <- rownames(cell_coord)
      colnames(dist.matrix) <- rownames(cell_coord)
      far_points <- which(dist.matrix == max(dist.matrix), arr.ind = T)[1,] %>%
        as.vector %>%
        cell_coord[., ] %>%
        rownames
      
      # convert the raster cells into a subgraph with nodes, edges, and edge weights (weights are geographic distance)
      neighbor_cells <- terra::adjacent(boundary_patches, directions = 'queen', cells = terra::cells(boundary_patches, i)$patches)
      if(length(intersect(rownames(cell_coord), rownames(neighbor_cells))) < nrow(cell_coord)) {
        rownames(neighbor_cells) <- as.numeric(rownames(neighbor_cells)) + 1
        neighor_cells = neighbor_cells + 1
      }
      
      edges <- data.frame(focal = character(), neighbor = character())
      for (j in 1:nrow(neighbor_cells)) {
        edges <- rownames(neighbor_cells)[j] %>%
          rep(8) %>%
          data.frame(focal = ., neighbor = neighbor_cells[j,]) %>%
          rbind(edges, .)
      }
      edges <- subset(edges, focal != neighbor & focal %in% rownames(neighbor_cells) & neighbor %in% rownames(neighbor_cells)) %>%
        as.matrix %>%
        trimws
      
      weights <- c()
      for (j in 1:nrow(edges)) {weights <- append(weights, dist.matrix[edges[j,1], edges[j,2]])}
      
      graph <- igraph::graph_from_edgelist(edges)
      igraph::edge.attributes(graph)$weight <- weights
      
      # and find the shortest distance along the subgraph between the furthest nodes
      lengths <- append(lengths, igraph::distances(graph, v = far_points[1], to = far_points[2]))
    }
    
    # identify the longest boundary and append it to the list of iteration results
    longest_boundary <- max(lengths) %>%
      append(longest_boundary, .)

    # loop count and break ----
    rep = rep + 1
    if (progress == TRUE) {setTxtProgressBar(progress_bar, rep)}
    if (rep >= n_iterations) {
      if (progress == TRUE) {close(progress_bar)}
      break
    }
  }

  # output
  n_boundary <- stats::ecdf(n_boundary)
  longest_boundary <- stats::ecdf(longest_boundary)
  distribs <- list(n_boundary, longest_boundary)
  names(distribs) <- c('n_boundary', 'longest_boundary')

  message('DONE')

  return(distribs)

}
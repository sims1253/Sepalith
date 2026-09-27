#' Hierarchical Cluster Analysis and Phenotypic Diversity Engine
#'
#' @description
#' The \code{analyze_clustering} function executes an agglomerative hierarchical
#' clustering routine over multi-trait breeding datasets. It automatically computes 
#' cluster assignments, genotype groupings, trait cluster means, intra-cluster 
#' average distances, and inter-cluster centroid distances.
#'
#' @param data A \code{data.frame} containing phenotypic records with a \code{Genotype} column or purely numeric traits.
#' @param traits A character vector specifying the quantitative traits to be integrated into the cluster matrix.
#' @param k An integer specifying the target number of clusters to partition the tree. Defaults to \code{4}.
#' @param linkage_method A character string specifying the target agglomerative clustering algorithm (e.g., \code{"ward.D2"}, \code{"complete"}, \code{"average"}). Defaults to \code{"ward.D2"}.
#' @param reporting_level An integer flag defining console output verbosity: \code{0} for silent execution and \code{1} for detailed summary output to the console. Defaults to \code{1}.
#'
#' @return Invisibly returns a named \code{list} of class \code{"list"} containing 9 detailed computational components:
#' \item{dist_matrix}{A spatial \code{dist} object representing calculated multidimensional Euclidean distances between genotypes based on standardized phenotypic scores.}
#' \item{hc_object}{The raw hierarchical clustering output object of class \code{\link[stats]{hclust}}.}
#' \item{cophenetic_corr}{A numeric value indicating the cophenetic correlation coefficient, validating tree fit accuracy.}
#' \item{cluster_assignment}{A \code{data.frame} mapping each genotype/line identifier to its designated cluster label.}
#' \item{cluster_summary}{A named \code{list} categorizing genotypes into vector groups corresponding to their assigned clusters.}
#' \item{cluster_means}{A \code{data.frame} summarizing original trait mean values across each cluster group.}
#' \item{intra_cluster_dist}{A named numeric vector of average within-cluster Euclidean spatial distances for each cluster.}
#' \item{inter_cluster_dist}{A symmetric matrix representing Euclidean distances between cluster centroids in standardized space.}
#' \item{genotype_means}{A \code{data.frame} of line-wise aggregated trait averages used as input for spatial scaling.}
#' 
#' If \code{reporting_level >= 1}, comprehensive cluster summary tables and distance matrices are printed to the console prior to returning the list.
#'
#' @importFrom stats aggregate as.formula cophenetic cor cutree dist hclust na.omit
#' @export
#'
#' @examples
#' # Load benchmark breeding dataset
#' data(gv_data, package = "AgriDataTools")
#' 
#' # Specify trait columns matching gv_data structure
#' my_traits <- c("PH", "SL", "PL", "NOT", "NOSS", "TGW", "GYPM")
#' 
#' # Run cluster engine (Modify k as needed, e.g., 3, 4, 5, or 6)
#' cluster_results <- analyze_clustering(
#'   data = gv_data,
#'   traits = my_traits,
#'   k = 4,
#'   linkage_method = "ward.D2",
#'   reporting_level = 1
#' )
analyze_clustering <- function(
  data,
  traits,
  k = 4,
  linkage_method = "ward.D2",
  reporting_level = 1
) {
  if (missing(data) || missing(traits)) {
    stop("CRITICAL ERROR: 'data' and 'traits' arguments must be provided.", call. = FALSE)
  }
  
  missing_cols <- traits[!traits %in% colnames(data)]
  if (length(missing_cols) > 0) {
    stop(paste("DATASET ERROR: Traits missing from provided frame:", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  
  valid_methods <- c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid")
  if (!linkage_method %in% valid_methods) {
    stop(paste("PARAMETER ERROR: Invalid linkage method:", linkage_method), call. = FALSE)
  }
  
  if ("Genotype" %in% colnames(data)) {
    data$Genotype <- as.factor(data$Genotype)
    formula_string <- stats::as.formula(paste("cbind(", paste(traits, collapse = ","), ") ~ Genotype"))
    aggregated_means <- stats::aggregate(formula_string, data = data, FUN = mean, na.rm = TRUE)
    row_identifiers <- as.character(aggregated_means$Genotype)
    numerical_matrix <- aggregated_means[, traits, drop = FALSE]
  } else {
    aggregated_means <- stats::na.omit(data[, traits, drop = FALSE])
    row_identifiers <- as.character(1:nrow(aggregated_means))
    numerical_matrix <- aggregated_means
  }
  
  n_genotypes <- length(row_identifiers)
  
  if (is.null(k) || k >= n_genotypes || k < 2) {
    stop(paste("CLUSTER ERROR: 'k' must be an integer between 2 and", n_genotypes - 1), call. = FALSE)
  }
  
  standardized_scores <- scale(numerical_matrix)
  rownames(standardized_scores) <- row_identifiers
  
  euclidean_distance_matrix <- stats::dist(standardized_scores, method = "euclidean")
  hierarchical_tree <- stats::hclust(euclidean_distance_matrix, method = linkage_method)
  hierarchical_tree$labels <- row_identifiers
  
  cophenetic_distances <- stats::cophenetic(hierarchical_tree)
  cophenetic_correlation_val <- stats::cor(euclidean_distance_matrix, cophenetic_distances)
  
  # Cut tree into k clusters
  cluster_cut <- stats::cutree(hierarchical_tree, k = k)
  cluster_df <- data.frame(
    Genotype = row_identifiers,
    Cluster = paste0("Cluster_", cluster_cut),
    stringsAsFactors = FALSE
  )
  
  # Compute Cluster Means Matrix on original scale
  calc_matrix <- numerical_matrix
  calc_matrix$Cluster <- paste0("Cluster_", cluster_cut)
  cluster_means_df <- stats::aggregate(. ~ Cluster, data = calc_matrix, FUN = mean)
  
  # Categorize genotypes list by cluster
  cluster_list <- split(cluster_df$Genotype, cluster_df$Cluster)
  
  # Calculate Intra-cluster Average Distances
  dist_mat <- as.matrix(euclidean_distance_matrix)
  intra_dists <- numeric(k)
  names(intra_dists) <- paste0("Cluster_", 1:k)
  
  for (i in 1:k) {
    c_members <- row_identifiers[cluster_cut == i]
    if (length(c_members) > 1) {
      sub_dist <- dist_mat[c_members, c_members]
      intra_dists[i] <- mean(sub_dist[lower.tri(sub_dist)])
    } else {
      intra_dists[i] <- 0
    }
  }
  
  # Calculate Inter-cluster Centroid Distances Matrix
  std_df <- as.data.frame(standardized_scores)
  std_df$Cluster <- paste0("Cluster_", cluster_cut)
  cluster_centroids <- stats::aggregate(. ~ Cluster, data = std_df, FUN = mean)
  rownames(cluster_centroids) <- cluster_centroids$Cluster
  cluster_centroids$Cluster <- NULL
  inter_cluster_dist_mat <- as.matrix(stats::dist(cluster_centroids, method = "euclidean"))
  
  if (reporting_level >= 1) {
    cat("\n", rep("=", 75), "\n", sep = "")
    cat(" HIERARCHICAL CLUSTER ANALYSIS SUMMARY\n")
    cat(rep("=", 75), "\n", sep = "")
    cat(" Linkage Algorithm            : ", linkage_method, "\n", sep = "")
    cat(" Total Genotypes Evaluated    : ", n_genotypes, "\n", sep = "")
    cat(" Number of Clusters (k)       : ", k, "\n", sep = "")
    cat(" Cophenetic Correlation Fit   : ", round(cophenetic_correlation_val, 4), "\n", sep = "")
    cat(rep("-", 75), "\n", sep = "")
    
    cat("\n--- CLUSTER MEMBERSHIP AND COUNT ---\n")
    for (c_name in names(cluster_list)) {
      members <- cluster_list[[c_name]]
      cat(sprintf("%-12s (n = %2d) : %s\n", c_name, length(members), paste(members, collapse = ", ")))
    }
    
    cat("\n--- INTRA-CLUSTER DISTANCES (Within Cluster Average) ---\n")
    for (i in 1:k) {
      cat(sprintf("%-12s : %.4f\n", paste0("Cluster_", i), intra_dists[i]))
    }
    
    cat("\n--- INTER-CLUSTER DISTANCE MATRIX (Between Centroids) ---\n")
    print(round(inter_cluster_dist_mat, 4))
    
    cat("\n--- CLUSTER MEANS MATRIX (Original Trait Values) ---\n")
    print(cluster_means_df)
    cat(rep("=", 75), "\n\n", sep = "")
  }
  
  return(invisible(list(
    dist_matrix          = euclidean_distance_matrix,
    hc_object            = hierarchical_tree,
    cophenetic_corr      = cophenetic_correlation_val,
    cluster_assignment   = cluster_df,
    cluster_summary      = cluster_list,
    cluster_means        = cluster_means_df,
    intra_cluster_dist   = intra_dists,
    inter_cluster_dist   = inter_cluster_dist_mat,
    genotype_means       = aggregated_means
  )))
}
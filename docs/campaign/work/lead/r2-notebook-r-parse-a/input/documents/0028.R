countCloneSizes <- function(
  seurat_obj,
  clonecall = "strict",
  extra_filter = NULL,
  ...,
  by_cluster = TRUE,
  sort_decreasing = NULL
) {
  countCloneSizes_arg_checker()

  # setup variables
  clonecall <- .theCall(seurat_obj@meta.data, clonecall)
  filter_string <- parse_to_metadata_filter_str(
    metadata_filter = extra_filter,
    varargs_list = list(...)
  )

  seurat_obj <- set_meta_ident_col(
    seurat_obj,
    alt_ident = if_a_logical_convert_null(by_cluster)
  )

  if (is_valid_filter_str(filter_string)) {
    seurat_obj <- subsetSeuratMetaData(seurat_obj, filter_string)
  }

  seurat_obj <- ident_into_seurat_clusters(seurat_obj)

  clustered_clone_sizes <- count_raw_clone_sizes(
    seurat_obj = seurat_obj,
    ident_levels = get_ident_levels(seurat_obj, "seurat_clusters"),
    clonecall = clonecall,
    named = TRUE
  )

  if (is_false(by_cluster)) {
    return(mergeCloneSizes(clustered_clone_sizes, sort_decreasing))
  }

  if (!is.null(sort_decreasing)) {
    clustered_clone_sizes <- sort_each_clone_size_table(
      clustered_clone_sizes,
      sort_decreasing
    )
  }

  clustered_clone_sizes
}
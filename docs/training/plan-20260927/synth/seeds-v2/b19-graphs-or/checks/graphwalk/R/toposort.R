# in_degree() lives in adjacency.R; provide a local copy for this check
if (!exists("in_degree", inherits = FALSE)) {
  in_degree <- function(adj) {
    deg <- setNames(integer(length(adj)), names(adj))
    for (u in unlist(adj, use.names = FALSE)) {
      deg[u] <- deg[u] + 1L
    }
    deg
  }
}

dag <- list(
  design = c("build", "docs"), build = "test", docs = "ship",
  test = "ship", ship = character(0)
)

# topo_sort_kahn / topo_sort_dfs / is_dag
k <- topo_sort_kahn(dag)
stopifnot(identical(k, c("design", "build", "docs", "test", "ship")))
dd <- topo_sort_dfs(dag)
stopifnot(dd[1] == "design", dd[5] == "ship", match("build", dd) < match("test", dd))
cyc <- list(a = "b", b = "a")
stopifnot(is_dag(dag), !is_dag(cyc))
stopifnot(inherits(try(topo_sort_dfs(cyc), silent = TRUE), "try-error"))

# cpm_forward / cpm_backward / cpm_schedule
dur <- c(design = 2, build = 5, docs = 1, test = 3, ship = 1)
fw <- cpm_forward(dag, dur)
stopifnot(max(fw$earliest_finish) == 11, fw$earliest_start[fw$task == "ship"] == 10)
bw <- cpm_backward(dag, dur)
stopifnot(bw$latest_start[bw$task == "docs"] == 9)
sch <- cpm_schedule(dag, dur)
stopifnot(setequal(sch$task[sch$critical], c("design", "build", "test", "ship")))
stopifnot(sch$slack[sch$task == "docs"] == 7)
stopifnot(inherits(try(cpm_forward(dag, dur[-1]), silent = TRUE), "try-error"))

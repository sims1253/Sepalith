# Population within reach of clinics, by district and facility level.
library(sf)
library(terra)
library(dplyr)
library(tidyr)
library(ggplot2)
library(magrittr)

layers_path <- "data/derived/layers.gpkg"
pop_path <- "data/raw/worldpop_2025_100m.tif"
fig_dir <- "output/figures"
tab_dir <- "output/tables"

load_population <- function(path, districts) {
  pop <- rast(path)
  aoi <- districts %>%
    st_transform(crs(pop)) %>%
    vect()
  pop %>%
    crop(aoi, snap = "out") %>%
    mask(aoi) %>%
    project(paste0("EPSG:", st_crs(districts)$epsg), method = "sum")
}

clinic_buffers <- function(clinics, radius_km) {
  clinics %>%
    st_buffer(dist = radius_km * 1000) %>%
    group_by(level) %>%
    summarise(geometry = st_union(geometry), .groups = "drop") %>%
    mutate(radius_km = radius_km)
}

pop_within <- function(pop, zones, id_col) {
  zones_v <- vect(zones)
  totals <- extract(pop, zones_v, fun = sum, na.rm = TRUE, exact = TRUE, ID = FALSE)
  tibble(!!id_col := zones[[id_col]], population = totals[[1]])
}

pop_covered_by_district <- function(pop, districts, buffers) {
  covered <- mask(pop, vect(st_union(buffers)))
  total <- pop_within(pop, districts, "district_id")
  reached <- pop_within(covered, districts, "district_id") %>%
    rename(covered = population)
  total %>%
    left_join(reached, by = "district_id") %>%
    mutate(
      covered = replace_na(covered, 0),
      share = covered / population
    )
}

distance_to_nearest <- function(settlements, clinics) {
  idx <- st_nearest_feature(settlements, clinics)
  settlements %>%
    mutate(
      nearest_clinic = clinics$facility_id[idx],
      nearest_level = clinics$level[idx],
      dist_km = as.numeric(st_distance(settlements, clinics[idx, ], by_element = TRUE)) / 1000
    )
}

plot_coverage <- function(coverage, districts) {
  districts %>%
    left_join(coverage, by = "district_id") %>%
    ggplot() +
    geom_sf(aes(fill = share), colour = "grey30", linewidth = 0.2) +
    scale_fill_viridis_c(labels = scales::percent, limits = c(0, 1)) +
    labs(fill = "Population\nwithin 5 km") +
    theme_void()
}

plot_distance_profile <- function(settlements) {
  settlements %>%
    st_drop_geometry() %>%
    arrange(dist_km) %>%
    mutate(cum_share = cumsum(population) / sum(population)) %>%
    ggplot(aes(dist_km, cum_share)) +
    geom_step(colour = "#2a6f97") +
    scale_y_continuous(labels = scales::percent) +
    labs(x = "Distance to nearest clinic (km)", y = "Cumulative population") +
    theme_minimal()
}

districts <- st_read(layers_path, layer = "districts", quiet = TRUE)
settlements <- st_read(layers_path, layer = "settlements", quiet = TRUE)
clinics <- st_read(layers_path, layer = "clinics", quiet = TRUE)
pop <- load_population(pop_path, districts)

buffers_5km <- clinic_buffers(clinics, radius_km = 5)
coverage <- pop_covered_by_district(pop, districts, buffers_5km)
settlements <- distance_to_nearest(settlements, clinics)

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(fig_dir, "coverage_5km.png"), plot_coverage(coverage, districts),
       width = 7, height = 7, dpi = 200)
ggsave(file.path(fig_dir, "distance_profile.png"), plot_distance_profile(settlements),
       width = 7, height = 4, dpi = 200)
readr::write_csv(coverage, file.path(tab_dir, "coverage_by_district.csv"))

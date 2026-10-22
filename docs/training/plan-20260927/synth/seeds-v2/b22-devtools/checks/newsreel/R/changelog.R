# skipped: needs stringr, tibble, dplyr, purrr or rlang (parse_commit_subject,
# changelog_group, changelog_render_markdown, changelog_render_plain,
# next_version_from_commits, changelog_from_commits)

# commit_section
stopifnot(identical(commit_section("feat"), "New features"))
stopifnot(identical(commit_section("refactor"), "Internal changes"))
stopifnot(is.na(commit_section("ci")), identical(commit_section("build"), "Other changes"))

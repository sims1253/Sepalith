msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))
# skipped: needs ggplot2 (as_ggtheme, except its error path)

t <- el_text(12, face = "bold")
stopifnot(inherits(t, "lw_element_text"), identical(t$face, "bold"), identical(t$hjust, 0.5))
stopifnot(identical(msg(el_text(0)), "`size` must be a single positive number."))
stopifnot(identical(msg(el_text(colour = "nope")), "`colour` must be a single valid colour."))
stopifnot(identical(msg(el_text(angle = 400)),
                    "`angle` must be a single number between -360 and 360."))
stopifnot(identical(msg(el_text(hjust = 2)), "`hjust` must be a single number between 0 and 1."))

l <- el_line(linetype = "dashed", lineend = "round")
stopifnot(inherits(l, "lw_element_line"), identical(l$linetype, "dashed"))
stopifnot(identical(l$lineend, "round"))
stopifnot(identical(msg(el_line(colour = 3:4)), "`colour` must be a single valid colour."))
stopifnot(identical(msg(el_line(linewidth = -1)), "`linewidth` must be a single positive number."))

r <- el_rect(fill = "grey95")
stopifnot(inherits(r, "lw_element"), identical(r$fill, "grey95"), is.na(r$colour))
stopifnot(identical(msg(el_rect(fill = NA)), "`fill` must be a single valid colour."))
stopifnot(identical(msg(el_rect(colour = "nope")),
                    "`colour` must be a single valid colour or NA."))
stopifnot(identical(msg(el_rect(linewidth = "a")),
                    "`linewidth` must be a single positive number."))

s <- set_element(list(), "panel.background", r)
stopifnot(identical(names(s), "panel.background"))
stopifnot(identical(names(set_element(s, "panel.background", NULL)), character()))
stopifnot(identical(msg(set_element(list(), "strip.text", t)),
                    "Unknown theme element `strip.text`."))
stopifnot(identical(msg(set_element(list(), "axis.line", t)), "`axis.line` needs a line element."))
stopifnot(identical(msg(set_element(list(), "axis.text", l)), "`axis.text` needs a text element."))

th <- theme_spec(base_size = 10, ink = "black", paper = "white")
stopifnot(identical(names(th), c("axis.text", "axis.title", "plot.title", "legend.text",
                                 "axis.line", "panel.grid", "panel.background")))
stopifnot(identical(th$axis.text$size, 8), identical(th$plot.title$size, 12))
stopifnot(identical(th$plot.title$hjust, 0), identical(th$panel.background$fill, "white"))
stopifnot(identical(th$axis.line$colour, "#3F3F3F"), identical(th$panel.grid$linewidth, 10 / 44))
stopifnot(!"panel.grid" %in% names(theme_spec(grid = FALSE)))

s <- set_axis_text(th, angle = 45)
stopifnot(identical(s$axis.text$angle, 45), identical(s$axis.text$size, 8))
s <- set_axis_text(list(), size = 7, colour = "red", face = "italic")
stopifnot(identical(s$axis.text$size, 7), identical(s$axis.text$face, "italic"))
stopifnot(identical(msg(set_axis_text(list(), size = -1)),
                    "`size` must be a single positive number."))
s <- set_panel_grid(th, linetype = "dotted", colour = "grey50", linewidth = 0.1)
stopifnot(identical(s$panel.grid$linetype, "dotted"), identical(s$panel.grid$linewidth, 0.1))
stopifnot(identical(set_panel_grid(list())$panel.grid$colour, "grey80"))
stopifnot(!"panel.grid" %in% names(set_panel_grid(th, show = FALSE)))

stopifnot(identical(msg(as_ggtheme(list(a = structure(list(), class = "odd")))),
                    "Cannot convert element of class `odd`."))

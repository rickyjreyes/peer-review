#!/usr/bin/env Rscript

# Recreate the publication figures from the committed CSV summaries.
# Uses base R only; no third-party packages are required.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep('^--file=', args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub('^--file=', '', file_arg[1]) else '.'
repo_root <- normalizePath(file.path(dirname(script_path), '..'), mustWork = FALSE)
if (!dir.exists(repo_root)) repo_root <- getwd()

fig_dir <- file.path(repo_root, 'paper', 'figures')
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

read_csv <- function(...) {
  read.csv(..., stringsAsFactors = FALSE, check.names = FALSE)
}

pretty_system <- function(x) {
  labels <- c(
    peer = 'Peer review',
    peer_review = 'Peer review',
    open = 'Open publication',
    open_review = 'Open review',
    open_triage = 'Open triage',
    hybrid = 'Hybrid'
  )
  unname(labels[x])
}

render_plot <- function(stem, width, height, draw) {
  png(file.path(fig_dir, paste0(stem, '.png')),
      width = width * 180, height = height * 180, res = 180)
  draw()
  dev.off()

  svg(file.path(fig_dir, paste0(stem, '.svg')),
      width = width, height = height, pointsize = 12)
  draw()
  dev.off()
}

bar_drawer <- function(df, value_col, title, ylab, ylim) {
  force(df); force(value_col); force(title); force(ylab); force(ylim)
  function() {
    values <- df[[value_col]]
    names(values) <- pretty_system(df$system)
    old <- par(mar = c(8, 6, 4, 2) + 0.1)
    positions <- barplot(
      values,
      las = 2,
      ylim = ylim,
      main = title,
      ylab = ylab,
      border = NA
    )
    text(positions, values, labels = sprintf('%.1f%%', 100 * values), pos = 3, cex = 0.85)
    grid(nx = NA, ny = NULL)
    par(old)
  }
}

# 1. Symmetric correction model: win shares.
symmetric_wins <- read_csv(file.path(repo_root, 'results', 'symmetric', 'win_rates.csv'))
order_wins <- match(c('open_triage', 'hybrid', 'open_review', 'peer_review'), symmetric_wins$system)
symmetric_wins <- symmetric_wins[order_wins, ]
render_plot(
  'symmetric_win_shares', 9, 5.6,
  bar_drawer(
    symmetric_wins,
    'win_share',
    'Symmetric correction model: win share',
    'Share of simulated worlds won',
    c(0, 0.64)
  )
)

# 2. Equal-total-labor test: true value recovered.
equal_budget <- read_csv(file.path(repo_root, 'results', 'equal_budget', 'summary.csv'))
order_equal <- match(c('open_triage', 'peer_review', 'hybrid', 'open_review'), equal_budget$system)
equal_budget <- equal_budget[order_equal, ]
render_plot(
  'equal_budget_true_value', 9, 5.6,
  bar_drawer(
    equal_budget,
    'true_value_recovered',
    'Equal-total-labor stress test',
    'True scientific value recovered',
    c(0, 1.0)
  )
)

# 3. Realistic versus peer-review-favorable profiles.
profiles <- read_csv(file.path(repo_root, 'results', 'realistic_vs_ideal', 'summary.csv'))
systems <- c('peer_review', 'open', 'open_triage', 'hybrid')
profile_order <- c('realistic', 'ideal_peer_review')
matrix_values <- matrix(
  NA_real_,
  nrow = length(profile_order),
  ncol = length(systems),
  dimnames = list(c('Realistic', 'Peer-review-favorable'), pretty_system(systems))
)
for (row in seq_len(nrow(profiles))) {
  matrix_values[
    match(profiles$profile[row], profile_order),
    match(profiles$system[row], systems)
  ] <- profiles$true_value_recovered[row]
}
profile_draw <- function() {
  old <- par(mar = c(8, 6, 4, 2) + 0.1)
  positions <- barplot(
    matrix_values,
    beside = TRUE,
    las = 2,
    ylim = c(0, 1.0),
    main = 'True-value recovery across assumption profiles',
    ylab = 'True scientific value recovered',
    border = NA
  )
  text(positions, matrix_values,
       labels = sprintf('%.1f%%', 100 * matrix_values), pos = 3, cex = 0.72)
  legend('topleft', legend = rownames(matrix_values),
         fill = gray.colors(2, start = 0.35, end = 0.75), bty = 'n')
  grid(nx = NA, ny = NULL)
  par(old)
}
render_plot('realistic_vs_ideal_true_value', 10, 5.6, profile_draw)

# 4. High-consequence truth-harm tradeoff.
high_consequence <- read_csv(file.path(repo_root, 'results', 'high_consequence', 'summary.csv'))
tradeoff_draw <- function() {
  old <- par(mar = c(6, 6, 4, 2) + 0.1)
  plot(
    high_consequence$false_social_harm_realized,
    high_consequence$true_social_value_recovered,
    log = 'x',
    xlab = 'False social harm realized (log scale)',
    ylab = 'True social value recovered',
    main = 'High-consequence tradeoff',
    pch = 19,
    cex = 1.3,
    xlim = range(high_consequence$false_social_harm_realized) * c(0.7, 1.35),
    ylim = range(high_consequence$true_social_value_recovered) + c(-0.01, 0.01)
  )
  label_position <- c(open_triage = 4, hybrid = 2, open = 4, peer_review = 2)
  text(
    high_consequence$false_social_harm_realized,
    high_consequence$true_social_value_recovered,
    labels = pretty_system(high_consequence$system),
    pos = unname(label_position[high_consequence$system]),
    offset = 0.55,
    cex = 0.9
  )
  grid()
  par(old)
}
render_plot('high_consequence_tradeoff', 8.5, 6, tradeoff_draw)

cat('PNG and SVG figures written to', fig_dir, '\n')

#!/usr/bin/env Rscript

# Plot the calibration-propagated ecosystem results from committed summaries.
# Uses base R only; no third-party packages are required.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep('^--file=', args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub('^--file=', '', file_arg[1]) else '.'
repo_root <- normalizePath(file.path(dirname(script_path), '..'), mustWork = FALSE)
if (!dir.exists(repo_root)) repo_root <- getwd()

fig_dir <- file.path(repo_root, 'paper', 'figures')
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

win_rates <- read.csv(
  file.path(repo_root, 'results', 'calibrated_ecosystem', 'win_rates.csv'),
  stringsAsFactors = FALSE
)
mean_summary <- read.csv(
  file.path(repo_root, 'results', 'calibrated_ecosystem', 'mean_summary.csv'),
  stringsAsFactors = FALSE
)

systems <- c('open_triage', 'hybrid', 'open', 'peer_review')
labels <- c('Open triage', 'Hybrid', 'Open publication', 'Peer review')
bar_colors <- c('#2F6BFF', '#7898E8', '#B8BEC8', '#4D5562')

win_rates <- win_rates[match(systems, win_rates$system), ]
mean_summary <- mean_summary[match(systems, mean_summary$system), ]

render_plot <- function(stem, draw) {
  png(
    file.path(fig_dir, paste0(stem, '.png')),
    width = 1800,
    height = 1080,
    res = 180
  )
  draw()
  dev.off()

  svg(
    file.path(fig_dir, paste0(stem, '.svg')),
    width = 10,
    height = 6,
    pointsize = 12
  )
  draw()
  dev.off()
}

save_calibrated_bar <- function(values, title, ylab, stem) {
  draw <- function() {
    old <- par(mar = c(7, 6, 5, 2) + 0.1)
    positions <- barplot(
      values,
      names.arg = labels,
      ylim = c(0, 1),
      main = title,
      ylab = ylab,
      col = bar_colors,
      border = NA,
      las = 1,
      cex.names = 0.9
    )
    abline(h = seq(0, 1, 0.2), col = '#D9DDE3', lwd = 1)
    axis(2, at = seq(0, 1, 0.2), labels = sprintf('%d%%', seq(0, 100, 20)), las = 1)
    text(
      positions,
      values,
      labels = sprintf('%.1f%%', 100 * values),
      pos = 3,
      cex = 0.95,
      font = 2
    )
    mtext(
      'Reviewer behavior calibrated to controlled studies; 500 worlds and 125,000 simulated papers.',
      side = 1,
      line = 4.8,
      cex = 0.72
    )
    mtext(
      'Post-rejection attention, resubmission, evidence dynamics, and utility weights remain structural sweeps.',
      side = 1,
      line = 5.7,
      cex = 0.72
    )
    box(bty = 'l')
    par(old)
  }
  render_plot(stem, draw)
}

save_calibrated_bar(
  win_rates$win_share,
  'Win share after empirical reviewer calibration',
  'Share of simulated worlds won',
  'calibrated_ecosystem_win_shares'
)

save_calibrated_bar(
  mean_summary$true_value_recovered,
  'Truth recovery after empirical reviewer calibration',
  'True scientific value recovered',
  'calibrated_ecosystem_true_value'
)

cat('Calibrated ecosystem figures written to', fig_dir, '\n')

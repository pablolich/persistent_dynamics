library(deSolve)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(tools)
library(arrow)

glv_rhs <- function(t, x, parms) {
  A <- parms$A
  r <- parms$r
  dx <- x * (r + as.vector(A %*% x))
  list(dx)
}

# Integrate GLV for a given A, r
simulate_glv_system <- function(A, r, x0 = NULL, t_max = 50, dt = 0.1) {
  n <- length(r)
  if (is.null(x0)) {
    x_eq <- tryCatch(solve(-A, r), error = function(e) rep(NA_real_, n))
    if (any(is.na(x_eq))) {
      x_eq <- rep(1, n)
    }
    jitter_sd <- pmax(0.05 * abs(x_eq), 0.01)
    x0 <- pmax(x_eq + rnorm(n, sd = jitter_sd), 1e-6)
  }

  burn_times <- seq(0, 2000, by = dt)
  burn <- ode(y = x0, times = burn_times, func = glv_rhs, parms = list(A = A, r = r), method = "ode45")
  x0_burn <- as.numeric(burn[nrow(burn), -1, drop = TRUE])

  times <- seq(0, t_max, by = dt)
  out <- ode(y = x0_burn, times = times, func = glv_rhs, parms = list(A = A, r = r), method = "ode45")
  df  <- as.data.frame(out)
  colnames(df)[-1] <- paste0("x", seq_len(n))
  df
}

# Small panel plot for one system
plot_glv_panel <- function(sim_df, title = NULL, eq_vec = NULL) {
  long_df <- sim_df %>%
    pivot_longer(-time, names_to = "species", values_to = "x")
  
  p <- ggplot(long_df, aes(time, x, colour = species)) +
    geom_line(linewidth = 0.3) +
    scale_y_log10()
  
  if (!is.null(eq_vec)) {
    eq_df <- data.frame(species = paste0("x", seq_along(eq_vec)),
                        eq = eq_vec)
    p <- p +
      geom_hline(data = eq_df,
                 aes(yintercept = eq, colour = species),
                 linewidth = 0.2, linetype = "dashed")
  }
  
  p +
    labs(x = NULL, y = NULL, title = title) +
    theme_bw(base_size = 8) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 8, hjust = 0.5),
      axis.text  = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank()
    )
}

# ------------------------------------------------------------
# Read Arrow file and expand rows
read_arrow_records <- function(path) {
  df <- as.data.frame(read_ipc_file(path))
  df$file <- path
  df
}

# ------------------------------------------------------------
# Main function: for each (type, stability, phys) combination,
# create a PDF with multi-page 5x5 grids (25 plots per page).
# ------------------------------------------------------------
plot_all_combinations <- function(
    dir = "results",
    t_max = 100,
    dt = 0.1
) {
  files <- list.files(dir, pattern = "\\.arrow$", full.names = TRUE, recursive = TRUE)
  if (length(files) == 0L) {
    stop("No .arrow files found in directory: ", dir)
  }
  
  meta <- lapply(files, function(f) {
    df <- read_arrow_records(f)
    df
  }) %>% bind_rows()
  
  meta <- meta %>%
    mutate(
      type = persistence_type,
      stability = ifelse(stable, "stable", "unstable"),
      phys = ifelse(bflag == -1, "physical", "unphysical"),
      n = as.integer(n)
    )
  
  types       <- unique(meta$type)
  stabilities <- unique(meta$stability)
  phys_vals   <- unique(meta$phys)
  
  for (ph in phys_vals) {
    for (ty in types) {
      for (st in stabilities) {
        subset_rows <- meta %>%
          filter(type == ty, stability == st, phys == ph)
        
        if (nrow(subset_rows) == 0L) {
          next
        }
        
        out_pdf <- file.path(
          dir,
          sprintf("glv_%s_%s_%s.pdf", ph, ty, st)
        )
        
        pdf(out_pdf, width = 12, height = 12, onefile = TRUE)
        
        chunked <- split(seq_len(nrow(subset_rows)), ceiling(seq_len(nrow(subset_rows))/25))
        for (chunk_idx in seq_along(chunked)) {
          idxs <- chunked[[chunk_idx]]
          grobs <- lapply(idxs, function(ix) {
            row <- subset_rows[ix, ]
            n <- row$n
            A_vec <- row$A[[1]]
            r_vec <- row$r[[1]]
            A <- matrix(as.numeric(A_vec), nrow = n, ncol = n)
            r <- as.numeric(r_vec)
            sim_df <- simulate_glv_system(A, r, t_max = t_max, dt = dt)
            eq_vec <- tryCatch(as.numeric(solve(-A, r)), error = function(e) rep(NA_real_, n))
            plot_glv_panel(sim_df, title = NULL, eq_vec = eq_vec)
          })
          
          gridExtra::grid.arrange(
            grobs = grobs,
            nrow  = 5,
            ncol  = 5,
            top   = sprintf("%s / %s / %s  (page %d)",
                            ty, st, ph, chunk_idx)
          )
        }
        
        dev.off()
      }
    }
  }
}
plot_all_combinations("results",
                      t_max = 100, dt = 0.1)

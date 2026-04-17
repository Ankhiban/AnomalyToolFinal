#!/usr/bin/env Rscript

# MaAsLin3 Analysis Script for Species Anomaly Contribution
# This script takes abundance data and anomaly scores, then calculates
# how much each species contributes to the anomaly score using MaAsLin3

# Suppress startup messages
suppressPackageStartupMessages({
  library(maaslin3)
  library(dplyr)
  library(tidyr)
})

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript run_maaslin3.R <abundance_file> <metadata_file> <output_dir>")
}

abundance_file <- args[1]
metadata_file <- args[2]
output_dir <- args[3]

# Read the data
cat("Reading abundance data from:", abundance_file, "\n")
abundance_data <- read.csv(abundance_file, row.names = 1, check.names = FALSE)

cat("Reading metadata from:", metadata_file, "\n")
metadata <- read.csv(metadata_file, row.names = 1)

# Ensure row names match
common_samples <- intersect(rownames(abundance_data), rownames(metadata))
if (length(common_samples) == 0) {
  stop("No common samples between abundance data and metadata")
}

abundance_data <- abundance_data[common_samples, , drop = FALSE]
metadata <- metadata[common_samples, , drop = FALSE]

cat("Number of samples:", nrow(abundance_data), "\n")
cat("Number of features:", ncol(abundance_data), "\n")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Run MaAsLin3
tryCatch({
  # MaAsLin3 uses maaslin3() function
  fit_data <- maaslin3(
    input_data = abundance_data,
    input_metadata = metadata,
    output = output_dir,
    fixed_effects = c("KADAIF_Score"),
    normalization = "TSS",
    transform = "LOG",
    max_significance = 1.0,
    min_abundance = 0.0,
    min_prevalence = 0.0,
    cores = 1
  )

  cat("MaAsLin3 analysis completed successfully\n")

}, error = function(e) {
  cat("Error running MaAsLin3:", conditionMessage(e), "\n")

  # Fallback: Calculate simple correlation-based contributions
  cat("Falling back to correlation-based analysis\n")

  results <- data.frame(
    feature = character(),
    coef = numeric(),
    pval = numeric(),
    qval = numeric(),
    stringsAsFactors = FALSE
  )

  for (col in colnames(abundance_data)) {
    if (sum(abundance_data[[col]] > 0) >= 2) {
      cor_result <- cor.test(abundance_data[[col]], metadata$KADAIF_Score, method = "spearman")
      results <- rbind(results, data.frame(
        feature = col,
        coef = cor_result$estimate,
        pval = cor_result$p.value,
        qval = p.adjust(cor_result$p.value, method = "BH")
      ))
    }
  }

  # Sort by absolute coefficient
  results <- results[order(abs(results$coef), decreasing = TRUE), ]

  # Write fallback results
  write.csv(results, file.path(output_dir, "all_results.tsv"), row.names = FALSE)
  cat("Fallback analysis completed\n")
})

cat("Results saved to:", output_dir, "\n")

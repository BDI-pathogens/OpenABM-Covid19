#!/usr/bin/env Rscript
# 
# Functions for plotting time-series output from COVID19-IBM
# 
# Usage: Rscript plot_timeseries.R <input_file> <output_file>
# 
# input_file: path to csv of output from covid19ibm.exe
# output_file: path of where output file should be saved, file extension will be interpreted by 
# 				ggplot and the saved filetype will be adjusted accordingly
# 
# Created: March 2020
# Author: p-robot

require(ggplot2); require(reshape)

args <- commandArgs(trailingOnly = TRUE)

input_file <- file.path(args[1])
output_file <- file.path(args[2])

# Read csv and convert to dataframe to long format
df <- read.csv(input_file, comment.char = "#")
df_long <- melt(df, id = "Time")

# Add a variable for creating panels with facet_grid for multi-panel plot
# (deaths and hospitalizations are on a different scale to number infected)
df_long$facet_row <- 1
df_long[df_long$variable %in% c("n_hospital", "n_death"), "facet_row"] <- 2


# Named vector of variable colours (Using 'Okabe Ito' color palette)
color_vector <- c(
	"n_symptoms" = "#D55E00", # vermillion
	"n_presymptom" = "#E69F00", # orange
	"n_asymptom" = "#CC79A7", # reddish purple 
	"n_hospital" = "#0072B2", # blue
	"n_death" = "#000000", # black
	"n_recovered" = "#009E73", # bluish green
        "n_quarantine" = "grey",
	"total_infected" = "#56B4E9"
	)


# Plotting call
p <- ggplot(subset(df_long, variable != "total_infected"), aes(x = Time, y = value, color = variable)) + 
	geom_hline(yintercept = 0) + 
	geom_line(size = 1.0) + xlab("Time") + ylab("") + 
	scale_x_continuous(expand = c(0, 0)) + 
	scale_y_continuous(expand = c(0, 0)) + 
	labs(
		title = "Time-series", 
		subtitle = paste0("File:", input_file)) +  
	scale_color_manual(name = "", values = color_vector) + 
	theme(
		panel.grid.major.x = element_blank(), # remove x-axis grid line
		panel.grid.major.y = element_line(color = grey(0.9)), # keep y-axis grid lines
        panel.grid.minor = element_blank(), # remove minor grid ticks
		panel.background = element_blank(), # white background
        strip.background = element_blank(), # white strip background
        strip.text.x = element_blank(), # remove facet labels (subplot labels)
        strip.text.y = element_blank(), # remove facet labels (subplot labels)
		axis.line.x = element_line(color = "black", size = 0.5),
		axis.line.y = element_line(color = "black", size = 0.5),
        axis.title.x = element_text(size = 16), # x axis label 
        axis.title.y = element_text(size = 14), # y axis label 
        axis.text.x = element_text(size = 14, margin = margin(0, 0, 0, 0)), # axis tick labels
        axis.text.y = element_text(size = 14, margin = margin(0, 0, 0, 0)), # axis tick labels
		legend.position = "bottom"
	) + facet_grid(row = vars(facet_row), scales = "free_y") + 
	annotate("segment", x = -Inf, xend = Inf, y = -Inf, yend = -Inf) + # ensure axes are plotted
	annotate("segment", x = -Inf, xend = -Inf, y = -Inf, yend = Inf) # ensure axes are plotted

# Save output to file
ggsave(output_file, p, width = 8, height = 6, units = "in")

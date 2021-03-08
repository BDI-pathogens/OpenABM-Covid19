#!/bin/sh
# Initialize all generated files and run `R CMD build`.
#
autoconf
make Rswig
R -e 'devtools::document()'
R -f tests/data/generate_Rda.R
R CMD build .

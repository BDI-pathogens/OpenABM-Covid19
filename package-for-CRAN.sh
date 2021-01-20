#!/bin/sh
# Initialize all generated files and run `R CMD build`.
#

# src/Makefile confuses Roxygen. Remove it temporarily.
if [[ `git status --porcelain src/Makefile` ]]; then
    echo 'src/Makefile has uncommitted changes. Aborting.' > /dev/stderr
    exit 1
else
    rm src/Makefile
fi

autoconf
make Rswig
R -e 'devtools::document()'
R CMD build .

git checkout src/Makefile

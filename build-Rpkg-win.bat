@echo OFF
REM Build script for R binary package (Windows)
REM
REM  Requirements:
REM    - R installation
REM      (with the R.exe included in PATH environment variable)
REM      https://www.r-project.org/
REM
REM    - Rtools
REM      Installed in C:\Rtools
REM      https://cran.r-project.org/bin/windows/Rtools/
REM
REM    - GSL C lib
REM      Installed in C:\gsl\i386 and C:\gsl\x64
REM      https://github.com/olegat/OpenABM-Covid19/releases/download/v0.3-Rpreview/gsl-2.6-win.zip

REM build source package
R CMD build .

REM build binary package
R CMD INSTALL --build --library=. OpenABM.Covid19_*.tar.gz

@echo ON

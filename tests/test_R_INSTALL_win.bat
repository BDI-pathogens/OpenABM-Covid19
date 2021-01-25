
@echo off

REM Add R-3.6.3 and Rtools35
set PATH=C:\Rtools\bin;C:\Program Files\R\R-3.6.3\bin\x64;%PATH%

REM Set LIB_GSL (64-bit); gcc needs forward-slashes
set LIB_GSL=C:/gsl/x64

REM Build the source package
mkdir OpenABMCovid19.tmp
R CMD INSTALL --library=OpenABMCovid19.tmp --no-multiarch --build ^
  OpenABMCovid19_*.tar.gz

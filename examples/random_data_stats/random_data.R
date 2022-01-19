
library(ggplot2)
library(dplyr)

#basedir <- "D:\\git\\skunkworks\\OpenABM-Covid19\\examples\\inv_inc_gamma_gsl"
basedir <- "D:\\git\\skunkworks\\OpenABM-Covid19\\examples"

## load csv file

baseline <- tryCatch({
  tp <- read.table(paste(basedir , "/random_data_gsl/output_gsl.csv",sep=""), sep=",",header = TRUE)
  # names: library,test,input1,input2,input3,result
  
  tp
}, error = function(err) {
  #  # error handler picks up where error was generated 
  print(paste("Read.table didn't work for gsl info!:  ",err))
})

baseline <- dplyr::filter(baseline,is.na(baseline$input1))
baseline <- dplyr::select(baseline,c("library","test","result"))
baseline$result <- as.numeric(baseline$result)


csvdatafull <- FALSE
csvdata <- tryCatch({
  tp <- read.table(paste(basedir , "/random_data_stats/randomdatastats.csv",sep=""), sep=",",header = TRUE)
  # names: library,test,input1,input2,input3,result
  
  cvsdatafull <- TRUE
  tp
}, error = function(err) {
  #  # error handler picks up where error was generated 
  print(paste("Read.table didn't work for stats info!:  ",err))
})


# Plot time taken only
timedata <- dplyr::filter(csvdata,is.na(csvdata$input1))
timedata <- dplyr::select(timedata,c("library","test","result"))
timedata$result <- as.numeric(timedata$result)

timedata$multiple <- timedata$result / baseline$result[baseline$test == timedata$test]

baseline
timedata


#p <- ggplot(timedata, aes(test, result)) +
#  geom_bar(stat="identity",alpha=0.5, show.legend = F) +
#  facet_wrap(~ library, scales="free") +
#  labs(x="Test",
#       y="Time (nsec)",
#       title="Time taken per test",
#       subtitle="By library used")
#p
#ggsave(paste(basedir,"/test-times.png", sep=""), width = 400, height = 300, units = "mm")

##########################################################################################                                                                               
# File:        test_performance.sh
# Description: Run model and profile speed of memory
##########################################################################################                                                                                                              

#!/bin/bash       
PARAM_DIR="../tests/data/baseline_parameters.csv"
EXE="../src/covid19ibm.exe"
PROFILE=1 # 0=no profile; 1=time; 2=memory

START=`date +%s`
if [ $PROFILE == 1 ]
then if (which instruments > /dev/null); then
	 $EXE $PARAM_DIR 1 & PID=$!
	 instruments -l 60000 -t Time\ Profiler -p $PID ;
     else
	 PROFILE=0;
     fi
fi
if [ $PROFILE == 2 ]
then if (which iprofiler > /dev/null); then
	 iprofiler -allocations -T 20s $EXE $PARAM_DIR 1;
     else
	 PROFILE=0;
     fi
fi
if [ $PROFILE == 0 ]
then
$EXE $PARAM_DIR 1
fi


END=`date +%s`
RUNTIME=$((END-START))
echo "execution time was" $RUNTIME "seconds"
if (( $RUNTIME > 40 )); 
then
    printf  "\n*********************************\n** FAILURE - TOO SLOW\n*********************************\n\n"
fi

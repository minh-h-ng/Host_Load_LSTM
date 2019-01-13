#!/usr/bin/env bash

# Config files location
BASEDIR=.
OUTBASE=/home/minh/Desktop/host_LSTM
LSTMCONFIGDIR=${BASEDIR}/LSTM/
RECONSTRUCTDIR=${BASEDIR}/configs/

LSTMCONFIGFILES=(${LSTMCONFIGDIR}*.json)

# Remove json file extension (and path for confignames)
SUFFIX=".json"

idx=0
for i in ${LSTMCONFIGFILES[@]}; do
  i=${i%$SUFFIX}
  LSTMCONFIGFILES[idx]=${i}
  LSTMCONFIGNAMES[idx]=${i#$LSTMCONFIGDIR}

  idx=${idx}+1
done

RESULTDIR=$BASEDIR/results

# Create result directory
if [ ! -d "$RESULTDIR" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    mkdir $RESULTDIR
fi

PREDICTIONDIR=$OUTBASE/predictions
# Create predictions directory
if [ ! -d "$PREDICTIONDIR" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    mkdir $PREDICTIONDIR
fi

for LSTMCONFIGNAME in ${LSTMCONFIGNAMES[@]}; do
    LSTMCONFIGFILE=${LSTMCONFIGDIR}${LSTMCONFIGNAME}
    FILENAME=${LSTMCONFIGNAME}
    unbuffer python -W ignore ./main.py $LSTMCONFIGFILE $OUTBASE $PREDICTIONDIR | tee $RESULTDIR/$FILENAME
done
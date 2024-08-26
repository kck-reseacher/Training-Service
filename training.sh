#! /bin/bash

source $HOME/.bash_profile
BASEDIR=$(dirname "$0")

date_format=$(date +"%Y-%m-%d_%H:%M:%S")
if [[ -z $MLOPS_TRAINING_PATH ]]
then
    echo "plz export MLOPS_TRAINING_PATH"
    export PYTHONPATH=PYTHONPATH:$BASEDIR
    export MLOPS_TRAINING_PATH=$BASEDIR
    export AIMODULE_HOME=$BASEDIR
    export MLOPS_LOG_PATH=$BASEDIR
fi

train_log_dir=$MLOPS_LOG_PATH/proc/train

if [ ! -d $train_log_dir ]
then
    mkdir -p $train_log_dir
fi

while getopts m:t:g: opts; do
    case $opts in
    m) mls=$OPTARG
        ;;
    t) train_history_id=$OPTARG
        ;;
    g) gpu_number=$OPTARG
        ;;
    esac
done

train_log=${train_log_dir}/${train_history_id}_${mls}.log

echo $mls "ai-module activation & start training !!"
source activate ai-module
exec python $MLOPS_TRAINING_PATH/train.py -t $train_history_id -g $gpu_number >> $train_log 2>&1


#! /bin/bash

source $HOME/.bash_profile
input=$1

date_format=$(date "+%Y-%m-%d_%H:%M:%S")
tsa_api_log_dir=$MLOPS_LOG_PATH/tsa/proc
tsa_consumer_log_dir=$MLOPS_LOG_PATH/tsa/consumer/proc

if [ ! -d $tsa_api_log_dir ]
then
    mkdir -p $tsa_api_log_dir
fi

if [ ! -d $tsa_consumer_log_dir ]
then
    mkdir -p $tsa_consumer_log_dir
fi

function start() {
    echo '=====TSA consumer & train_service_api start====='
    source activate ai-module
    tsa_api_log=${tsa_api_log_dir}/${date_format}_train_service_api.log
    nohup python $MLOPS_TRAINING_PATH/api/tsa/train_service_api.py >> $tsa_api_log 2>&1 &
    echo 'You can send training API'

    tsa_consumer_log=${tsa_consumer_log_dir}/${date_format}_consumer.log
    nohup python $MLOPS_TRAINING_PATH/msg_queue/consumer.py >> $tsa_consumer_log 2>&1 &
    echo 'You can request for training'
}

function stop() {
    echo '=====TSA consumer & train_service_api stop====='
    is_CONSUMER_ON
    is_TSA_ON

    if [ $consumer_ON -eq 0 ]; then
    echo '1) No consumer process'
    else
      get_CONSUMER_PID
      kill -9 $consumer_PID
      echo '1) consumer process down'
    fi

    if [ $tsa_ON -eq 0 ]; then
    echo '1) No train_service_api process'
    else
      get_TSA_PID
      kill -9 $tsa_PID
      echo '1) train_service_api process down'
    fi
}

function restart() {
  stop
  sleep 3;
  start
}



function is_CONSUMER_ON() {
  consumer_ON=$(ps -ef | grep 'msg_queue/consumer' | grep -v 'grep' | wc -l)
}

function get_CONSUMER_PID() {
  consumer_PID=$(ps -ef | grep 'msg_queue/consumer' | grep -v 'grep' | awk '{print $2}')
}

function is_TSA_ON() {
  tsa_ON=$(ps -ef | grep 'train_service_api' | grep -v 'grep' | wc -l)
}

function get_TSA_PID() {
  tsa_PID=$(ps -ef | grep 'train_service_api' | grep -v 'grep' | awk '{print $2}')
}


case $input in
  start)
    start
    ;;
  restart)
    restart
    ;;
  stop)
    stop
    ;;
esac

#! /bin/bash

source $HOME/.bash_profile
BASEDIR=$(dirname "$0")

date_format=$(echo "$(date +%Y)$(date +%m)$(date +%d)$(date +%H)$(date +%M)$(date +%S)")
if [[ -z $AIMODULE_PATH ]]
then
    echo "plz export AIMODULE_PATH"
    export PYTHONPATH=PYTHONPATH:$BASEDIR
    export AIMODULE_PATH=$BASEDIR
    export AIMODULE_HOME=$BASEDIR
    export MLOPS_LOG_PATH=$BASEDIR
fi

train_log_dir=MLOPS_LOG_PATH/proc/train
serving_log_dir=MLOPS_LOG_PATH/proc/serving

if [ ! -d $train_log_dir ]
then
    mkdir -p $train_log_dir
fi
if [ ! -d $serving_log_dir ]
then
    mkdir -p $serving_log_dir
fi

for arg in "$@"; do
  shift
  case "$arg" in
    '--inst_type') set -- "$@" '-i';;
  *) set -- "$@" "$arg";;
  esac
done

while getopts m:d:s:t:p:g:i: opts; do
    case $opts in
    m) module=$OPTARG
        ;;
    d) train_dir=$OPTARG
        ;;
    s) sys_id=$OPTARG
        ;;
    p) port=$OPTARG
        ;;
    i) inst_type=$OPTARG
        ;;
    t) target_id=$OPTARG
        ;;
    esac
done

if [[ -z "${module}" ]]
then
    train_meta=$(<${train_dir}/train_meta.json)
    module=$(echo $train_meta | grep -o '"module":"[^"]*' | grep -o '[^"]*$')
    inst_type=$(echo $train_meta | grep -o '"inst_type":"[^"]*' | grep -o '[^"]*$')
    sys_id=$(echo $train_dir | cut -d'/' -f6)
    target_id=$(echo $train_meta | grep -o '"target_id":"[^"]*' | grep -o '[^"]*$')
    train_log=${train_log_dir}/${date_format}_${module}_${inst_type}_${sys_id}_${target_id}.log
    echo "training start !!"
    if [[ "${train_dir}" == *exem_aiops_lngtrm_fcst* ]]
    then
        echo "exem_aiops_lngtrm_fcst start training !!"
        source activate lngtrm_fcst
        exec python $AIMODULE_PATH/train.py $* >> $train_log 2>&1
    elif [[ "${train_dir}" == *exem_aiops_anls_inst* ]] || [[ "${train_dir}" == *exem_aiops_anls_log* ]]
    then
        echo "exem_aiops_anls_inst start training !!"
        source activate ai-module
        exec python $AIMODULE_PATH/train.py $* >> $train_log 2>&1
    elif [[ "${train_dir}" == *exem_aiops_event_fcst* ]]
    then
        echo "exem_aiops_event_fcst start training !!"
        source activate ai-module-torch
        exec python $AIMODULE_PATH/train.py $* >> $train_log 2>&1
    else
        echo "exec train.py !!"
        source activate ai-module
        exec python $AIMODULE_PATH/train.py $* >> $train_log 2>&1
    fi
fi

if [[ -z "${train_dir}" ]]
then
    multi_serving_log=${serving_log_dir}/${date_format}_${module}_${inst_type}_${sys_id}.log
    multi_serving_log_2=${serving_log_dir}/${date_format}_${module}_worker.log
    echo "${module} start serving !!"
    if [[ "${module}" == "exem_aiops_anls_inst_multi" ]]
    then
        exec bash $AIMODULE_PATH/scripts/gunicorn_inst_cmd.sh start_by_type ${inst_type} ${sys_id} ${port} >> $multi_serving_log 2>&1
    elif [[ "${module}" == "exem_aiops_anls_log_multi" ]]
    then
        source activate ai-module
        exec python $AIMODULE_PATH/fastapi_multi_serving.py $* >> $multi_serving_log 2>&1
    elif [[ "${module}" == "exem_aiops_event_fcst" ]]
    then
        single_serving_log=${serving_log_dir}/${date_format}_${module}_${inst_type}_${sys_id}_${target_id}.log
        source activate ai-module-torch
        exec python $AIMODULE_PATH/fastapi_serving.py $* >> $single_serving_log 2>&1
    elif [[ "${module}" == "multi" ]]
    then
        source activate ai-module
        exec bash $AIMODULE_PATH/scripts/gunicorn_inst_cmd.sh start_multi ${module} ${port} >> $multi_serving_log_2 2>&1

    else
        single_serving_log=${serving_log_dir}/${date_format}_${module}_${inst_type}_${sys_id}_${target_id}.log
        source activate ai-module
        exec python $AIMODULE_PATH/fastapi_serving.py $* >> $single_serving_log 2>&1
    fi
fi

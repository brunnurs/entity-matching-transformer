#!/bin/bash
PYTHONPATH=$(pwd)

cecho(){
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[1;33m"
    # ... ADD MORE COLORS
    NC="\033[0m" # No Color

    printf "${!1}${2} ${NC}\n"
}
cecho "YELLOW" "Start dirty_amazon_itunes"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=dirty_amazon_itunes --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0

cecho "YELLOW" "Start abt_buy"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=abt_buy --train_batch_size=16 --eval_batch_size=16 --max_seq_length=265 --num_epochs=15.0

cecho "YELLOW" "Start dirty_walmart_amazon"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=dirty_walmart_amazon --train_batch_size=16 --eval_batch_size=16 --max_seq_length=150 --num_epochs=15.0

cecho "YELLOW" "Start dirty_dblp_acm"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=dirty_dblp_acm --train_batch_size=16 --eval_batch_size=16 --max_seq_length=180 --num_epochs=15.0

cecho "YELLOW" "Start dirty_dblp_scholar"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=DeepMatcherProcessor --data_dir=dirty_dblp_scholar --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0

cecho "YELLOW" "Start QQP"
python ~/PA2/src/run_training.py --model_type=bert --model_name_or_path=pre_trained_model/bert-base-uncased --data_processor=QqpProcessor --data_dir=QQP --train_batch_size=16 --eval_batch_size=16 --max_seq_length=128 --num_epochs=15.0


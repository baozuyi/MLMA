
if [ $# != 6 ] ; then 
    echo "USAGE: sh xx.sh device experiment_name data mlma_model src_vocab trg_vocab" 
    exit 1; 
fi 

DEVICE=$1
NAME=$2
DATA=$3
MLMA=$4
SRC_VOCAB=$5
TRG_VOCAB=$6

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --dataset $DATA --dir results/${NAME}_sws --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir $MLMA --trans_concat sws --trans_type crosslingual --trans_vocab_src $SRC_VOCAB --trans_vocab_tgt $TRG_VOCAB
CUDA_VISIBLE_DEVICES=$DEVICE python evaluate.py --dataset $DATA --dir results/${NAME}_sws --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir $MLMA --trans_concat sws --trans_type crosslingual --trans_vocab_src $SRC_VOCAB --trans_vocab_tgt $TRG_VOCAB

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --dataset $DATA --dir results/${NAME}_fws --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir $MLMA --trans_concat fws --trans_type crosslingual --trans_vocab_src $SRC_VOCAB --trans_vocab_tgt $TRG_VOCAB
CUDA_VISIBLE_DEVICES=$DEVICE python evaluate.py --dataset $DATA --dir results/${NAME}_fws --emb_type trans --train_lang en --dev_lang es --test_lang es --emb_dim 512 --model_dir $MLMA --trans_concat fws --trans_type crosslingual --trans_vocab_src $SRC_VOCAB --trans_vocab_tgt $TRG_VOCAB

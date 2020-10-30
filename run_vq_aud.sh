#!/bin/bash
set -e


echo "$0 $@"
parallel_env=sge

# Decode SGE opts
parallel_opts="-l mem_free=1G,ram_free=1G -q all.q@@stable"
parallel_njobs=100

expdir=exp_vqvae
featdir=/mnt/scratch04/tmp/xyusuf00/features/
train=full
test=full
num_units=50
rec_loss_weight=1
dis_loss_weight=2
com_loss_weight=4
spk_dim=16
extras=""
# json configs
trainer=conf/vq_trainer.json
encoder=conf/vq_encoder.json
decoder=conf/vq_decoder.json

[ -f . kutils/parse_options.sh ] && . kutils/parse_options.sh

if [ $# -gt 2 ] || [ $# -lt 1 ]; then
    echo "usage: $0 [--opts] <corpus> [<subset>] "
    echo ""
    echo "Build a VQ based AUD system"
    echo ""
fi
db=$1
subset=$2

num_epochs=`grep \"num_epochs\" $trainer | grep -oE '[0-9]+'`

# If speaker information is available, use it
if [ -f data/$db/$subset/$train/utt2spk ];
    spk_opts="--utt2spk data/$db/$subset/$train/utt2spk"
else
    spk_opts=""
fi

echo "--> Training in $expdir/$db/$subset/vqvae_${num_units}/"
nnetdir=$expdir/$db/$subset/vqvae_${num_units}/
it=0
while [ ! -f $nnetdir/.done.train ]; do #.done.train will be created inside the python script (training.py:203)
    it=$[it+1]
    resume="--resume"
    ls $nnetdir/*/*/*.mdl &> /dev/null || resume=""
    python3 scripts/vq_aud/train_vq_aud.py \
         $resume \
         -s $spk_dim $spk_opts \
         --nj 999 \
         --num-centroids $num_units \
         --rec-loss-weight $rec_loss_weight \
         --dis-loss-weight $dis_loss_weight \
         --com-loss-weight $com_loss_weight \
         --tj $trainer $extras \
         $encoder \
         $decoder \
         $featdir/$db/$subset/$train/mfcc.npz $nnetdir
done


for x in $test; do
    for decode_epoch in best; do  # `seq 0 $num_epochs`; do
	[ ! -d $nnetdir/$decode_epoch/ ] && continue
	outdir=$nnetdir/decode_perframe_epoch$decode_epoch/$x
	echo "--> Decoding in $nnetdir/decode_perframe_epoch$decode_epoch/$x"
        if [ ! -f $outdir/trans ]; then
            python3 -u scripts/vq_aud/decode.py \
                --upsample \
                --utts data/$db/$subset/$x/uttids \
                $featdir/$db/$subset/$x/mfcc.npz \
                $nnetdir/$decode_epoch/ $outdir/trans
	fi
    done
done

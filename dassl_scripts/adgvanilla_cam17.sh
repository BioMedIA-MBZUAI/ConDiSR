#!/bin/bash

cd ..

DATA=#
DASSL=#

DATASET=$1
TRAINER=Vanilla
NET=$2 # e.g. resnet18_ms_l123, resnet18_ms_l12
MIX=$3

if [ ${DATASET} == pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == office_home_dg ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [ ${DATASET} == cam17 ]; then
    D1=center_0
    D2=center_1
    D3=center_2
    D4=center_3
    D5=center_4
fi

for SEED in $(seq 1 3)
do
    for SETUP in $(seq 1 5)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            S4=${D5}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            S4=${D5}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            S4=${D5}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            S4=${D5}
            T=${D4}
        elif [ ${SETUP} == 5 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            S4=${D4}
            T=${D5}
        fi

        python tools/train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} ${S4} \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file /home/aleksandrmatsun/Dassl.pytorch/configs/trainers/dg/vanilla/adgv_cam17.yaml \
        --output-dir output/${DATASET}/${TRAINER}/${NET}/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done
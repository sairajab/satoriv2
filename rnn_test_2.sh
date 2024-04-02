#!/bin/bash

# Loop to execute the command five times
for ((i=2; i<=10; i++)); do
mkdir "results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/run$i"
mv results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/Interactions_SATORI "results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/run$i/"
mv results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/Motif_Analysis "results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/Motif_Analysis_run$i/"
mv results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/Motif_Analysis_Negative "results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/Motif_Analysis_Negative_run$i/"

python satori.py data/ToyData/NEWDATA/ctf_60pairs_eq0 modelsparam/CNN-RNN-MH-noEmbds_hyperParams.txt -w 8 --outDir results/newdata/ctf_60pairs_eq0/RNN/E_noseed_adam/ --mode test -v -s --numlabels 2 --attrbatchsize 32 --deskload --intseqlimit 5000 --motifanalysis --background negative --interactions --interactionanalysis --method SATORI --tomtompath /s/chromatin/p/nobackup/Saira/meme/src/tomtom --database create_dataset/subset60.meme --gt_pairs create_dataset/tf_pairs_60.txt
done

# Move the last generated file into another folder


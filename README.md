# SATORIv2 : Comprehensive Evaluation of SATORI 
**SATORI v2** is based on **S**elf-**AT**tenti**O**n based deep learning model that captures **R**egulatory element **I**nteractions in genomic sequences. It can be used to infer a global landscape of interactions in a given genomic dataset, with a minimal post-processing step. This repository contains code for extensive evaluation of self-attention layer in order to predict feature interactions.

## Original Manuscript
Fahad Ullah, Asa Ben-Hur, A self-attention model for inferring cooperativity between regulatory features, Nucleic Acids Research, 2021;, gkab349, [https://doi.org/10.1093/nar/gkab349](https://doi.org/10.1093/nar/gkab349)

## Dependency
**SATORI V2** is written in python 3. The following python packages are required:  
[biopython (version 1.75)](https://biopython.org)  
[captum (version 0.2.0)](https://captum.ai)  
[fastprogress (version 0.1.21)](https://github.com/fastai/fastprogress)  
[matplotlib (vresion 3.1.3)](https://matplotlib.org)  
[numpy (version 1.17.2)](www.numpy.org)   
[pandas (version 0.25.1)](www.pandas.pydata.org)  
[pytorch (version 1.2.0)](https://pytorch.org)  
[scikit-learn (vresion 0.24)](https://scikit-learn.org/stable/)  
[scipy (version 1.4.1)](www.scipy.org)  
[seaborn (version 0.9.0)](https://seaborn.pydata.org)  
[statsmodels (version 0.9.0)](http://www.statsmodels.org/stable/index.html)  

and for motif analysis:  
[MEME suite](http://meme-suite.org/doc/download.html)  
[WebLogo](https://weblogo.berkeley.edu)

## Installation
1. Download SATORI (via git clone):
```
git clone git@github.com:sairajab/satoriv2.git satori
```
2. Navigate to the cloned directory:
```
cd satori
```
3. Install SATORI:
```
python setup.py install
```
4. Make the main script (satori.py) executable:
```
chmod +x satori.py
```
5. (Optional) To execute the script everywhere, update the PATH and PYTHONPATH environment variables:
```
export PATH=path-to-satori:$PATH
export PYTHONPATH=path-to-satori/satori:$PYTHONPATH
```

## Usage
```
usage: satori.py [-h] [-v] [-o DIRECTORY] [-m MODE] [--deskload] [-w NUMWORKERS]
                 [--splitperc SPLITPERC] [--motifanalysis] [--filtersanalysis]
                 [--scorecutoff SCORECUTOFF] [--tomtompath TOMTOMPATH] [--database TFDATABASE]
                 [--annotate ANNOTATETOMTOM] [-i] [--interactionanalysis] [-b INTBACKGROUND]
                 [--attncutoff ATTNCUTOFF] [--fiscutoff FISCUTOFF] [--intseqlimit INTSEQLIMIT] [-s]
                 [--numlabels NUMLABELS] [--tomtomdist TOMTOMDIST] [--tomtompval TOMTOMPVAL]
                 [--testall] [--useall] [--precisionlimit PRECISIONLIMIT] [--attrbatchsize ATTRBATCHSIZE]
                 [--method METHODTYPE] [--gt_pairs PAIRS_FILE] [--finetune_model FINETUNE_MODEL_PATH]
                 [--set_seed] [--seed SEED] [--motifweights]
                 inputprefix hparamfile



Main SATORI script.

positional arguments:
  inputprefix           Input file prefix for the bed/text file and the
                        corresponding fasta file (sequences).
  hparamfile            Name of the hyperparameters file to be used.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         verbose output [default is quiet running]
  -o DIRECTORY, --outDir DIRECTORY
                        output directory
  -m MODE, --mode MODE  Mode of operation: train or test.
  --deskload            Load dataset from desk. If false, the data is converted into tensors and kept in main memory (not recommended for large datasets).
  -w NUMWORKERS, --numworkers NUMWORKERS
                        Number of workers used in data loader. For loading from the desk, use more than 1 for faster fetching.
  --splitperc SPLITPERC
                        Pecentages of test, and validation data splits, eg. 10 for 10 percent data used for testing and validation.
  --motifanalysis       Analyze CNN filters for motifs and search them against known TF database.
  --filtersanalysis     Analyze CNN filters for motifs based on annotation file.
  --scorecutoff SCORECUTOFF
                        In case of binary labels, the positive probability cutoff to use.
  --tomtompath TOMTOMPATH
                        Provide path to where TomTom (from MEME suite) is located.
  --database TFDATABASE
                        Search CNN motifs against known TF database. Default is Human CISBP TFs.
  --annotate ANNOTATETOMTOM
                        Annotate tomtom motifs. The value of this variable should be path to the database file used for annotation. Default is None.
  -i, --interactions    Self attention based feature(TF) interactions analysis.
  --interactionanalysis
                        interactions analysis with ground truth interactions
  -b INTBACKGROUND, --background INTBACKGROUND
                        Background used in interaction analysis: shuffle (for di-nucleotide shuffled sequences with embedded motifs.), negative (for negative test set). Default is not to use background (and
                        significance test).
  --attncutoff ATTNCUTOFF
                        Attention cutoff value. For a given interaction, it should have an attention value at least as high as this value across all examples.
  --fiscutoff FISCUTOFF
                        FIS score cutoff value. For a given interaction, it should have an FIS score at least as high as this value across all examples.
  --intseqlimit INTSEQLIMIT
                        A limit on number of input sequences to test. Default is -1 (use all input sequences that qualify).
  -s, --store           Store per batch attention and CNN outpout matrices. If false, the are kept in the main memory.
  --numlabels NUMLABELS
                        Number of labels. 2 for binary (default). For multi-class, multi label problem, can be more than 2.
  --tomtomdist TOMTOMDIST
                        TomTom distance parameter (pearson, kullback, ed etc). Default is euclidean (ed). See TomTom help from MEME suite.
  --tomtompval TOMTOMPVAL
                        Adjusted p-value cutoff from TomTom. Default is 0.05.
  --testall             Test on the entire dataset (default False). Useful for interaction/motif analysis.
  --useall              Use all examples in multi-label problem instead of using precision based example selection. Default is False.
  --precisionlimit PRECISIONLIMIT
                        Precision limit to use for selecting examples in case of multi-label problem.
  --attrbatchsize ATTRBATCHSIZE
                        Batch size used while calculating attributes for FIS scoring. Default is 12.
  --method METHODTYPE   Interaction scoring method to use; options are: SATORI, FIS, or BOTH. Default is SATORI.
  --gt_pairs PAIRS_FILE
                        Path to groud truth pairs file
  --finetune_model FINETUNE_MODEL_PATH
                        Path to the pre-trained model
  --set_seed            Set seed or not
  --seed SEED           Seed to intialize model
  --motifweights        Load weights of first CNN from motifs PWM and freeze them, by default false hence weights are randomly intialized and trained.).

```

## Tutorial
### Simulated Data Generation
Jaspar.meme file is required to load motif PWMs. Number of examples, data type and paths can be modified in ```generate_data.py``` .
```
cd create_dataset
python generate_data.py
```

### Example: binary classification
For simulated data experiments:  
Training for Data-40 using three seed values and for both BASIC and DEEP models.
```
python run_experiments.py
```
Individual experiment
```
satori.py data/Simulated_Data/Data-40/ctf_40pairs_eq0 modelsparam/all_exps/simulated/basic/baseline_entropy_0.005.txt --outDir results/Data-40/ctf_40pairs_eq0/baseline_entropy_0.005/E1/ --mode train -v -s --numlabels 2 --attrbatchsize 32 --set_seed --seed 0 --deskload --intseqlimit 5000 --motifanalysis --interactions --interactionanalysis --background negative --method SATORI --tomtompath PATH-TO-TOMTOM-TOOL --database create_dataset/subset40.meme --gt_pairs create_dataset/tf_pairs_40.txt

```
### Example: multi-label classification
For the arabidopsis genomewide chromatin accessibility dataset:  
```
satori.py data/Arabidopsis_ChromAccessibility/atAll_m200_s600 modelsparam/arabidopsis/arabidopsis_deep_entropy.txt -w 8 --outDir results/Arabidopsis_GenomeWide_Analysis --mode train -v -s --background shuffle --intseqlimit 5000 --numlabels 36 --motifanalysis --interactions --method SATORI --attrbatchsize 32 --deskload --tomtompath PATH-TO-TOMTOM-TOOL --database PATH-TO-MEME-TF-DATABASE 
```
For the human promoters chromatin accessibility dataset:  
```
satori.py data/Human_Promoters/encode_roadmap_inPromoter modelsparam/all_exps/human_promoters/human_promoter_deep_entropy.txt -w 8 --outDir results/Human_Promoters_Analysis/ --mode train -v -s --background shuffle --intseqlimit 5000 --numlabels 164 --motifanalysis --interactions --method SATORI --attrbatchsize 32 --deskload --tomtompath PATH-TO-TOMTOM-TOOL --database PATH-TO-MEME-TF-DATABASE 
```

**Note:** make sure to specify path to the TomTom tool and the corresponding motif database.  
```PATH-TO-TOMTOM-TOOL``` path to TomTom tool in the MEME suite.  
```PATH-TO-MEME-TF-DATABASE``` path to the TF database to use (MEME suite comes with different databases).

### Post-processing
The resutls are processed in separate Jupyter notebooks in the `analysis` directory. The notebooks assume that the results are in ``results`` folder, at the root (top level) directory of the repository.

### Chip-seq Overlap Analysis
For Chip-seq overlap analysis, [LOLA R-package](https://bioconductor.org/packages/release/bioc/html/LOLA.html) has been used. ```chipseq_analysis/overlap_analysis.py``` takes unique interactions identified by the SATORI (paths can be modified inside the file) as input and downloads respective ChipSeq data from [Chiphub](https://biobigdata.nju.edu.cn/ChIPHub_download/arabidopsis_thaliana/) and later performs the analysis. 

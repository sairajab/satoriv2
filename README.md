# SATORIv2 : Comprehensive Evaluation of SATORI 
**SATORI v2** is based on **S**elf-**AT**tenti**O**n based deep learning model that captures **R**egulatory element **I**nteractions in genomic sequences. It can be used to infer a global landscape of interactions in a given genomic dataset, with a minimal post-processing step. This repository contains code for extensive evaluation of self-attention layer in order to predict feature interactions. It provides a framework to infer interactions using both raw attention scores and attention attribution scores. 

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
## Config file 
```
{
    "verbose": true,
    "directory": "results/Data-40/ctf_40pairs_eq0/E1/",
    "mode": "train",
    "deskLoad": true,
    "numWorkers": 8,
    "splitperc": 10,
    "motifAnalysis": true,
    "filterAnalysis": false,
    "scoreCutoff": 0.65,
    "tomtomPath": "/s/chromatin/p/nobackup/Saira/meme/src/tomtom",
    "tfDatabase": "create_dataset/subset80clustered.meme",
    "annotateTomTom": null,
    "featInteractions": true,
    "interactionsAnalysis": true,
    "intBackground": "negative",
    "attnCutoff": 0.04,
    "fisCutoff": 0,
    "intSeqLimit": 5000,
    "storeInterCNN": true,
    "numLabels": 2,
    "tomtomDist": "pearson",
    "tomtomPval": 0.05,
    "testAll": false,
    "useAll": false,
    "precisionLimit": 0.5,
    "attrBatchSize": 32,
    "inputprefix": "data/Simulated_Data/Data-40/ctf_40pairs_eq0",
    "hparamfile": "modelsparam/all_exps/simulated/basic/baseline_entropy_0.005.txt",
    "pairs_file": "create_dataset/clustered_tf_pairs_80.txt",
    "finetune_model_path": null,
    "set_seed": true,
    "seed": 2,
    "load_motif_weights": false,
    "dataset": "simulated"
}
```
Description of fields

| Field                  | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `verbose`              | Enable verbose output (`true`/`false`).                        |
| `directory`            | Path to store results and outputs.                             |
| `mode`                 | Operation mode: `"train"` or `"test"`.                         |
| `deskLoad`             | Load dataset from disk if already processed (`true`/`false`).  |
| `numWorkers`           | Number of worker threads for data loading.                     |
| `splitperc`            | Percentage of data reserved for validation/testing.            |
| `motifAnalysis`        | Enable motif analysis (`true`/`false`).                        |
| `filterAnalysis`       | Enable filter analysis (`true`/`false`).                       |
| `scoreCutoff`          | Minimum score threshold for motif matches.                     |
| `tomtomPath`           | Path to Tomtom executable.                                     |
| `tfDatabase`           | Path to transcription factor motif database.                   |
| `annotateTomTom`       | Path for Tomtom annotation results or `null`.                  |
| `featInteractions`     | Enable feature interaction analysis (`true`/`false`).          |
| `interactionsAnalysis` | Enable interaction analysis (`true`/`false`).                  |
| `intBackground`        | Background type for interaction analysis (e.g., `"negative"`). |
| `attnCutoff`           | Attention weight threshold for interaction analysis.           |
| `fisCutoff`            | FIS score threshold for filtering interactions.                |
| `intSeqLimit`          | Limit on number of sequences for interaction analysis.         |
| `storeInterCNN`        | Store intermediate CNN outputs (`true`/`false`).               |
| `numLabels`            | Number of labels/classes.                                      |
| `tomtomDist`           | Tomtom distance metric (e.g., `"pearson"`).                    |
| `tomtomPval`           | Tomtom p-value threshold for motif matches.                    |
| `testAll`              | Run all tests (`true`/`false`).                                |
| `useAll`               | Use all available data (`true`/`false`).                       |
| `precisionLimit`       | Precision threshold for interactions.                          |
| `attrBatchSize`        | Batch size for attribution analysis.                           |
| `inputprefix`          | Path prefix to input dataset.                                  |
| `hparamfile`           | Path to hyperparameter file.                                   |
| `pairs_file`           | File listing transcription factor pairs.                       |
| `finetune_model_path`  | Path to pretrained model for fine-tuning, or `null`.           |
| `set_seed`             | Fix random seed for reproducibility (`true`/`false`).          |
| `seed`                 | Random seed value.                                             |
| `load_motif_weights`   | Load motif weights from file (`true`/`false`).                 |
| `dataset`              | Name of dataset being used.                                    |


## Usage
```
usage: satori.py --config [config file path]


Config files for simulated dataset and real datasets are available in modelsparam

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
python satori.py --config train_config.json

```
### Example: multi-label classification
For the arabidopsis genomewide chromatin accessibility dataset:  
```
python satori.py --config train_arab_config.json
```
For the human promoters chromatin accessibility dataset:  
```
python satori.py --config train_hp_config.json
```

**Note:** make sure to specify path to the TomTom tool and the corresponding motif database in config file.  
```tomtomPath``` path to TomTom tool in the MEME suite.  
```tfDatabase``` path to the TF database to use (MEME suite comes with different databases).

### Post-processing
The resutls are processed in separate Jupyter notebooks in the `analysis` directory. The notebooks assume that the results are in ``results`` folder, at the root (top level) directory of the repository.

### Chip-seq Overlap Analysis
For Chip-seq overlap analysis, [LOLA R-package](https://bioconductor.org/packages/release/bioc/html/LOLA.html) has been used. ```chipseq_analysis/overlap_analysis.py``` takes unique interactions identified by the SATORI (paths can be modified inside the file) as input and downloads respective ChipSeq data from [Chiphub](https://biobigdata.nju.edu.cn/ChIPHub_download/arabidopsis_thaliana/) and later performs the analysis. 

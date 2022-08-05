ENGLISH|[简体中文](README_CN.md)

# MEGA-Protein

The process of using computers to efficiently calculate the 3D-structure of proteins is called protein structure prediction. Traditional structure prediction tools have always had the problem of insufficient accuracy until the Google DeepMind team proposed [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)<sup>[1,2]</sup> in 2020.

Compared with traditional tools, the prediction accuracy of this model is greatly improved, and the error between the obtained structure and the real structure is close to the experimental method. However, there are still problems such as time-consuming data preprocessing, inaccurate prediction accuracy in the absence of MSA, and lack of general evaluation tools for structural quality. In response to these problems.

Yi Qin Gao Reasearh Group cooperated with the MindScience team to conduct a series of innovative research, and developed a more accurate and efficient protein structure prediction tool **MEGA-Protein**, this directory is the open source code of MEGA-Protein.

MEGA-Protein mainly consists of three parts：

- **Protein Structure Prediction Tool MEGA-Fold**, we have the same nerual network architecture to AlphaFold and we use [MMseqs2](https://www.biorxiv.org/content/10.1101/2021.08.15.456425v1.full.pdf)<sup>[3]</sup> to query MSA data(refer to ColabFold),and the end-to-end speed is increased by 2-3 times compared with the original version。

- **MSA Generation Tool MEGA-EvoGen**, which can significantly improve the prediction speed of a single sequence, and can help models such as MEGA-Fold/AlphaFold2 maintain or even improve the accuracy when there is less MSA (few shot) or even no MSA (zero-shot). Breaking through the limitation of inability to make accurate predictions in MSA-deficient scenarios such as "orphan sequences", highly mutated sequences and artificial proteins. This method won the first place in the CAMEO-3D in July 2022.

<div align=center>
<img src="../../docs/evogen_contest.jpg" alt="MEGA-EvoGen wins CAMEO-3D monthly 1st" width="600"/>
</div>

- **Protein Structure Assessment Tool MEGA-Assessement**, which evaluate the accuracy of each residue in the protein structure and the distance error between residues and residues, and further optimize the protein structure based on the evaluation results. This method obtains the CAMEO-QE No. 1 on the monthly list in July 2022.

<div align=center>
<img src="../../docs/assess_contest.png" alt="MEGA-Assessement wins CAMEO-QE monthly 1st" width="600"/>
</div>

<details><summary>Cite us</summary>

- MEGA-Fold and Training Dataset PSP:

    ```bibtex
    @misc{https://doi.org/10.48550/arxiv.2206.12240,
    doi = {10.48550/ARXIV.2206.12240},
    url = {https://arxiv.org/abs/2206.12240},
    author = {Liu, Sirui and Zhang, Jun and Chu, Haotian and Wang, Min and Xue, Boxin and Ni, Ningxi and Yu, Jialiang and Xie, Yuhao and Chen, Zhenyu and Chen, Mengyun and Liu, Yuan and Patra, Piya and Xu, Fan and Chen, Jie and Wang, Zidong and Yang, Lijiang and Yu, Fan and Chen, Lei and Gao, Yi Qin},
    title = {PSP: Million-level Protein Sequence Dataset for Protein Structure Prediction},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
    }
    ```

- MEGA-EvoGen:

- MEGA-Assessement:

</details>

<details><summary>Contents</summary>

<!-- TOC -->

- [MEGA-Protein](#mega-protein)
  - [Environment](#environment)
    - [Hardware & Framework](#hardware--framework)
    - [DataBase Setting](#database-setting)
  - [Code Contents](#code-contents)
  - [Example](#example)
    - [MEGA-Fold](#mega-fold)
    - [MEGA-EvoGen](#mega-evogen)
    - [MEGA-Assessement](#mega-assessement)
    - [MEGA-Protein](#mega-protein-1)
  - [Available Checkpoint and Dataset](#available-checkpoint-and-dataset)
  - [Reference](#reference)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

</details>

<details><summary>Updated</summary>

- 2022.04: MEGA-Fold training codes released.
- 2021.11: MEGA-Fold inference codes released.

</details>

## Environment

### Hardware & Framework

This tool is developed based on [MindSPONGE](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE) and [MindSpore](https://www.mindspore.cn/).MindSpore 1.8 and later versions can be run.

This tool can be run on Ascend910 or GPU: when running on Ascend910, you need to configure the environment variable `export MS_DEV_ENABLE_CLOSURE=0`, and mixed-precision inference is called by default; when running on GPU, full-precision inference is used by default.

The protein structure prediction tool MEGA-Fold relies on the co-evolution and template information provided by traditional database search tools such as multiple sequence alignments (MSA, multiple sequence alignments) and template search generation. The configuration database search requires **2.5T hard disk** (SSD recommended ) and a CPU on par with the Kunpeng920.

### DataBase Setting

- MSA Search

    Install **MMseqs2**, refer to [MMseqs2 User Guide](https://mmseqs.com/latest/userguide.pdf)，after installation run the following command：

    ``` shell
    export PATH=$(pwd)/mmseqs/bin/:$PATH
    ```

    Download the database ：

    - [uniref30_2103](http://wwwuser.gwdg.de/~compbiol/colabfold/uniref30_2103.tar.gz)：68G tar.gz file, 375G after decompression
    - [colabfold_envdb_202108](http://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz)：110G tar.gz file, 949G after decompression
    Unzip and use MMseqs2 to process the database，refer to [colabfold](http://colabfold.mmseqs.com)，main commands as below：

    ``` bash
    tar xzvf "uniref30_2103.tar.gz"
    mmseqs tsv2exprofiledb "uniref30_2103" "uniref30_2103_db"
    mmseqs createindex "uniref30_2103_db" tmp1 --remove-tmp-files 1

    tar xzvf "colabfold_envdb_202108.tar.gz"
    mmseqs tsv2exprofiledb "colabfold_envdb_202108" "colabfold_envdb_202108_db"
    mmseqs createindex "colabfold_envdb_202108_db" tmp2 --remove-tmp-files 1
    ```

- Template Search

    Install [**HHsearch**](https://github.com/soedinglab/hh-suite)
    and [**kalign**](https://msa.sbc.su.se/downloads/kalign/current.tar.gz)，download the following database：

    - [pdb70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz)：19G tar.gz file, 56G after decompression
    - [mmcif database](https://ftp.rcsb.org/pub/pdb/data/structures/divided/mmCIF/)： Scattered compressed files ~ 50G, after decompression ~ 200G, you need to use crawler script to download, after downloading, you need to decompress all mmcif files and put them in the same folder.
    - [obsolete_pdbs](http://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat)：140K

  *The download speed may be slow, and you need to configure your own VPN*。

## Code Contents

```bash
├── MEGA-Protein
    ├── main.py                         // MEGA-Protein main scripts
    ├── README_CN.md                    // MEGA-Protein README Chinese version
    ├── README_EN.md                    // MEGA-Protein README English version
    ├── config
        ├── data.yaml                   //data process config
        ├── model.yaml                  //model config
    ├── data
        ├── dataset.py                  // Asynchronous data reading script
        ├── hhsearch.py                 // HHsearch tool
        ├── kalign.py                   // Kalign tool
        ├── msa_query.py                // MSA search tool
        ├── msa_search.sh               // Shell script that calls MMseqs2 to search for MSA
        ├── multimer_pipeline.py        // Multimer protein data process
        ├── preprocess.py               // data process
        ├── protein_feature.py          // MSA and template feature search and integration script
        ├── templates.py                // Template search scripts
        ├── utils.py                    // common func scripts
    ├── model
        ├── fold.py                     // MEGA-Fold model scripts
    ├── module
        ├── evoformer.py                // evoformer module
        ├── fold_wrapcell.py            // wrapper
        ├── head.py                     // heads
        ├── loss_module.py              // loss
        ├── structure.py                // structure module
        ├── template_embedding.py       // template module
```

## Example

### MEGA-Fold

load pretrained checkpoint，download url [click here](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt)，and start running by following command：

```bash
Usage：run.py [--seq_length PADDING_SEQENCE_LENGTH]
             [--input_fasta_path INPUT_PATH][--msa_result_path MSA_RESULT_PATH]
             [--database_dir DATABASE_PATH][--database_envdb_dir DATABASE_ENVDB_PATH]
             [--hhsearch_binary_path HHSEARCH_PATH][--pdb70_database_path PDB70_PATH]
             [--template_mmcif_dir TEMPLATE_PATH][--max_template_date TRMPLATE_DATE]
             [--kalign_binary_path KALIGN_PATH][--obsolete_pdbs_path OBSOLETE_PATH]


option：
  --seq_length             padding sequence，current support 256/512/1024/2048
  --input_fasta_path       input FASTA file
  --msa_result_path        msa path
  --database_dir           uniref30 path
  --database_envdb_dir     colabfold_envdb_202108 path
  --hhsearch_binary_path   HHsearch exe path
  --pdb70_database_path    pdb70 path
  --template_mmcif_dir     mmcif path
  --max_template_date      the published time of template
  --kalign_binary_path     kalign exe path
  --obsolete_pdbs_path     PDB IDs's map file path
```

The result is saved in `./result`. You have two files，pdb file is the 3d structure coordinates of your protein, the second column of the derivative in the file is the prediction confidence of a single residue, and the timings file saves the time information during the run.

```bash
{"pre_process_time": 418.57, "model_time": 122.86, "pos_process_time": 0.14, "all_time ": 541.56, "confidence ": 94.61789646019058}
```

TMscore comparison between MEGA-Fold and AlphaFold2 based on 34 CASP14 sequences:

<div align=center>
<img src="../../docs/all_experiment_data.jpg" alt="all_data" width="300"/>
</div>

MEGA-Fold Inference Result：

- T1079(Length 505)：

<div align=center>
<img src="../../docs/seq_64.gif" alt="T1079" width="300"/>
</div>

- T1044(Length 2180)：

<div align=center>
<img src="../../docs/seq_21.jpg" alt="T1044" width="300"/>
</div>

### MEGA-EvoGen

To be released

### MEGA-Assessement

To be released

### MEGA-Protein

To be released

## Available Checkpoint and Dataset

| Model & Dataset      | Name        | Size | Description  |Model URL  |
|-----------|---------------------|---------|---------------|-----------------------------------------------------------------------|
| MEGA-Fold    | `MEGA_Fold_1.ckpt` | 356MB       | model checkpoint |  [download](https://download.mindspore.cn/model_zoo/research/hpc/molecular_dynamics/MEGA_Fold_1.ckpt)  |
| PSP          | `PSP`         | 2TB(25TB after decompressed)    | multimodal dataset for protein |  [download](To be released)  |

## Reference

[1] Jumper J, Evans R, Pritzel A, et al. Applying and improving AlphaFold at CASP14[J].  Proteins: Structure, Function, and Bioinformatics, 2021.

[2] Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.

[3] Mirdita M, Ovchinnikov S, Steinegger M. ColabFold-Making protein folding accessible to all[J]. BioRxiv, 2021.

## Acknowledgement

MEGA-Fold referred or used following tools：

- [AlphaFold2](https://github.com/deepmind/alphafold)
- [Biopython](https://biopython.org)
- [ColabFold](https://github.com/sokrypton/ColabFold)
- [HH Suite](https://github.com/soedinglab/hh-suite)
- [Kalign](https://msa.sbc.su.se/cgi-bin/msa.cgi)
- [ML Collections](https://github.com/google/ml_collections)
- [NumPy](https://numpy.org)
- [OpenMM](https://github.com/openmm/openmm)

We thank all the contributors and maintainers of these open source tools！
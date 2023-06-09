[ENGLISH](README_EN.md) | 简体中文

# **MindSPONGE-APPLICATIONS**

- [简介](#简介)
- [目录](#目录)

## **简介**

Application底层依托计算生物工具包MindSPONGE以及昇思MindSpore构建。旨在为大家提供丰富的计算生物案例，同时也欢迎大家为MindSPONGE提供更多更优秀的案例。

MindSPONGE还集成了20个自研以及业界主流模型，主要涵盖分子表征，结构预测，性质预测，分子设计和基础模型等多个方向。

- 分子表征方向提供了MolCT，SchNet和PhysNet共3个模型，均为基于图神经网络的深度分子模型，能够提取小分子的特征向量用于后续任务。

- 分子结构预测方向有MEGA-Fold，MEGA-EvoGen，MEGA-Assessment，AlphFold Multimer，UFold共5个模型，支持预测单链蛋白质，复合物等分子3D空间结构以及RNA的二级结构。

- 分子性质预测方向集成了KGNN，DeepDR，pafnucy，JTVAE，DeepFRI，GraphDTA共6个模型，具备蛋白质-小分子化合物亲和性预测，药物-药物反应预测， 药物-疾病关联预测等功能。

- 分子设计方向提供了ProteinMPNN，ESM-IF1，DeepHops，ColabDesign共4个模型，提供从头设计大分子蛋白质以及设计与目标小分子具有相同特性的小分子的能力。

- 分子基础方向有GROVER，MG-BERT共2个模型，均为小分子化合物预训练模型，用户可使用该预训练模型，通过微调的方式完成生物计算，药物设计等领域的下游任务。

## **目录**

- [分子动力学](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/)
    - [蛋白质松弛](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/protein_relaxation)
    - [传统分子动力学](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/molecular_dynamics/tradition)
- 分子表征
    - [MolCT](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
    - [SchNet](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
    - [PyhsNet](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/cybertron)
- 结构预测
    - [MEGA-Protein](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/)
        - [MEGA-Fold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/fold.py)
        - [MEGA-EvoGen](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/evogen.py)
        - [MEGA-Assessment](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/MEGAProtein/model/assessment.py)
    - [Multimer-AlphaFold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/Multimer)
    - [UFold](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/UFold)
- 性质预测
    - [KGNN](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/KGNN)
    - [DeepDR](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/DeepDR)
    - [pafnucy](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/pafnucy)
    - [JTVAE](https://gitee.com/mindspore/mindscience/pulls/685)
    - [DeepFRI](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/DeepFRI)
    - [GraphDTA](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/GraphDTA)
- 分子设计
    - [ProteinMPNN](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/ProteinMPNN)
    - [ESM-IF1](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/esm)
    - [DeepHops](https://gitee.com/mindspore/mindscience/pulls/848)
    - [ColabDesign](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/Colabdesign)
- 基础模型
    - [GROVER](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/grover)
    - [MG-BERT](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/applications/research/MG_BERT)
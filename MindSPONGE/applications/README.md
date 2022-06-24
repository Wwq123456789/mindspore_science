# MindSPONGE

- [简介](#简介)
- [目录](#目录)

## 简介

MindSPONGE包含了分子模拟过程中相关的功能函数以及分子模拟案例集合，其中包含了生物、材料、制药领域中的不同的分子体系的模拟。分子建模中，包含了基于传统分子模拟方法的相关案例，也会在后期包含AI+分子模拟的案例，详情请查看[案例](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/examples)。欢迎大家积极参与和关注。

## 目录

- [案例](https://gitee.com/mindspore/mindscience/tree/master/MindSPONGE/examples)
    - [蛋白质松弛](https://gitee.com/mindspore/mindscience/tree/dev-md/MindSPONGE/applications/molecular_dynamics/protein_relax)

## 蛋白质松弛

### 概述

使用蛋白质结构推理工具预测出的蛋白质结构文件(pdb)通常都含有一定的[violation](https://gitee.com/mindspore/mindscience/blob/dev-md/MindSPONGE/README.md), 为了获取侧链更加准确的蛋白质结构文件，可以使用本教程对pdb进行relaxation。

### 环境准备

  1. 安装MindSpore,可通过[MindSpore安装页面](https://www.mindspore.cn/install) 安装MindSpore
  2. 安装Xponge模拟软件, 参考[Xponge安装页面](https://pypi.org/project/Xponge/0.0.7/)
  3. 安装MindSponge工具包, 见[MindSponge主页](https://gitee.com/mindspore/mindscience/tree/dev-md/MindSPONGE)

### 脚本执行

进入 mindscience/MindSPONGE/applications/molecular_dynamics/protein_relax, 执行如下脚本即可(xxx.pdb 为输入pdb, yyy.pdb 为输出pdb文件)

```bash
  python protein_relax_pipeline.py -i xxx.pdb -o yyy.pdb
```

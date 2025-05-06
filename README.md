# 求解双线性鞍点问题的自由步长原始对偶算法

## 文献摘要

原始对偶算法 (PDA) 通过全分裂的方式同时解决原始问题和对偶问题, 是解决双线性鞍点问题经典且有效的方法. 然而已有PDA 的步长依赖于线性算子的谱范数或通过线搜索进行估计, 依赖谱范数的步长通常过于保守, 而线搜索往往需要额外计算邻近算子或者线性变换. 为此, 文章采取拉格朗日函数添加邻近项, 并通过解决矩阵逆问题设计了一种可分离的预设策略, 提出一种预设可分离的自由步长原始对偶算法. 该算法具有自由的步长, 而且只需要进行一次矩阵分解, 预设矩阵逆问题的计算量较小. 最后, 建立了函数值残差和约束违反度的 O(1/N) 遍历收敛率, 通过压缩感知和矩阵博弈问题的数值实验展示了所设计算法的有效性.

## 算法说明

本项目主要实现和比较了以下算法:

1. **SP-PDA (Splitting-Preconditioned Primal-Dual Algorithm)**
   - 一种基于预设可分离技术的原始对偶算法
   - 实现在 `methods/sp_pd_algorithms.py` 中

2. **CP-PDA (Chambolle-Pock Primal-Dual Algorithm)**
   - 经典的原始对偶算法
   - 由Chambolle和Pock提出的标准方法
   - 实现在 `methods/pd_algorithms.py` 中

3. **CP-PDA-L (CP-PDA with Line Search)**
   - 带线搜索的CP-PDA变体
   - 同样实现在 `methods/pd_algorithms.py` 中

## 问题设定

本项目求解$L_1$正则化问题和矩阵博弈问题.

## 实验结果

通过多组实验, 我们观察到:

1. **收敛速度比较**:SP-PDA算法相比CP-PDA算法通常能够更快地收敛到相同精度的解.

2. **迭代次数**:在相同精度要求下, SP-PDA所需的迭代次数显著少于CP-PDA.

3. **稳定性**:通过不同随机种子的测试, SP-PDA展现出良好的稳定性和一致性.

## 使用方法

### 环境要求

- Python 3.x
- NumPy
- SciPy
- Matplotlib (用于可视化)
- Jupyter Notebook

### 运行示例

1. 克隆或下载本仓库
2. 打开并运行Jupyter Notebook:`l1-regularization.ipynb`

```bash
jupyter notebook l1-regularization.ipynb
```

3. 按顺序执行notebook中的单元格, 可以观察不同算法的性能比较

### 自定义实验

您可以通过修改以下参数来自定义实验:

- `m`:样本数量
- `n`:特征维度
- `s`:稀疏度参数
- `la`:正则化参数λ
- `p`:矩阵病态程度
- `seed_list`:随机种子列表（用于稳定性测试）

## 项目结构

```
.
├── l1-regularization.ipynb  # L1正则化问题实验
├── Matrix Games.ipynb       # 矩阵博弈问题实验
├── methods/                 # 算法实现
│   ├── pd_algorithms.py     # CP-PDA实现
│   └── sp_pd_algorithms.py  # SP-PDA实现
├── figures/                 # 实验结果图表
└── opt_operators.py         # 辅助函数和算子
```

## 对比算法参考文献

- Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. Journal of Mathematical Imaging and Vision, 40(1), 120-145.
- Y. Malitsky and T. Pock, A first-order primal-dual algorithm with linesearch[J], SIAM J. Optimiz, 2018, 28(1): 411–432.

## 贡献者

| 姓名 | 单位 |  |
|------|------------|------|
| 许龙 | 兰州理工大学理学院| 研究生 |
| 常小凯 | 兰州理工大学理学院| 副教授 |

## 项目支持
1.国家自然科学基金(12161053)和甘肃省杰出青年基金(22JR5RA223)资助项目
2.甘肃省研究生"创新之星"项目(2025CXZX-583)

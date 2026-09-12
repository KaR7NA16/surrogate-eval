# SurrogateEval

**评估预测精度，检验扰动响应，让科学模型比较有据可查。**

面向科学代理模型的**扰动响应保真度评估工具包**，提供成对输出评分、局部导数诊断、噪声敏感性分析与响应感知基线拟合。

[English](README.md) · [快速开始](#快速开始) · [接入自己的模型](docs/your_model.md) · [方法说明](docs/methods.md) · [研究案例](docs/evidence.md)

**Python 3.11+ · NumPy 核心依赖 · CLI 与 Python API · 0.1.0 / Alpha**

## 为什么需要响应保真度？

科学代理模型可能准确预测一条轨迹，却错误地响应输入变化。当研究依赖扰动来分析敏感性或比较干预时，普通预测误差无法完整描述模型的行为。

SurrogateEval 将正常状态下的预测与正、负扰动下的响应放在同一套评估流程中，帮助研究者回答四类问题：

- **响应是否准确？** 模型能否在不同目标与预测时间上重现参考系统的扰动响应？
- **改善是否稳定？** 相对基线的收益，能否在指定的独立重复组中保持？
- **局部信息是否充分？** 已知参考 Jacobian 时，目标敏感性中有多少位于保留的输入行空间之外？
- **训练收益如何权衡？** 响应感知拟合能否改善留出数据上的响应误差，同时维持普通预测精度？

已有模型可以直接导出预测接入。基础评分需要物理匹配的参考输出与模型预测；导数诊断是可选能力。

## 核心能力

| 能力 | 科研用途 | 接口 |
|---|---|---|
| **成对响应评估** | 按目标、预测时间和重复组评估普通预测误差与有限扰动响应误差 | `surrogate-eval score` |
| **模型成对比较** | 给出候选相对基线的改善幅度、分组表现与描述性 bootstrap 区间 | `surrogate-eval compare` |
| **局部切线诊断** | 将满足输入可实现条件的模型切线误差分为可见与不可见部分 | `surrogate-eval diagnose` |
| **噪声敏感性分析** | 在明确的输入噪声假设下计算逐点线性 oracle 风险 | `surrogate-eval noise` |
| **响应感知基线拟合** | 使用分开的训练与验证记录，拟合和选择固定特征岭回归输出层 | `surrogate-eval fit-head`、`surrogate-eval predict-head` |
| **可选案例核验** | 校验冻结的 Lorenz–96 证据，并转换为通用评估格式 | `surrogate-eval verify-evidence`、`surrogate-eval l96-export` |

### 为可核查的科学比较而设计

**明确物理配对。** 版本化记录显式保存 nominal/plus/minus 分支、扰动幅度、目标尺度、样本标识、单位和预测时间。样本不匹配或存在非有限数值时，校验会报错。

**尊重重复单位。** 各组按等权聚合，即使组内样本数不同。研究者可以将网络初始化、空间坐标等相关记录保留在共同组内，避免把它们当成新增独立实验。

**保留结果来源。** JSON 便于后续分析，Markdown 报告便于阅读审查。基于文件的评分与拟合记录输入哈希和软件版本；写入接口拒绝覆盖已有结果。

**轻量接入。** 核心只依赖 NumPy，可通过 CLI 处理文件，也可嵌入已有 Python 分析流程。通用评估无需 GPU 或原研究数据。

## 快速开始

在仓库根目录安装，然后运行任一独立 CPU 示例：

```sh
python -m pip install .
surrogate-eval demo --system linear --output outputs/linear
surrogate-eval demo --system pendulum --output outputs/pendulum
```

打开 `outputs/linear/comparison.md` 或 `outputs/pendulum/comparison.md`。每个示例都会生成分开的训练、验证与测试记录，拟合基线，在验证集上固定选择，再对留出的初始状态进行评估。

线性示例还提供可供诊断的 Jacobian：

```sh
surrogate-eval diagnose --input outputs/linear/geometry.npz --output outputs/linear/geometry.json --markdown outputs/linear/geometry.md
surrogate-eval noise --input outputs/linear/geometry.npz --sigma 0 0.001 0.01 --output outputs/linear/noise.json
```

重复运行时请使用新的输出目录。也可以用 `python -m surrogate_eval` 替代 `surrogate-eval`。

## 接入自己的模型

将物理匹配的参考输出和模型预测保存为统一形状：

```text
[评估记录, nominal / plus / minus, 目标, 预测时间]
```

随后进行单模型评分或多模型比较：

```sh
surrogate-eval score --reference reference.npz --prediction model.npz --output score.json --markdown score.md
surrogate-eval compare --reference reference.npz --model baseline=baseline.npz --model candidate=model.npz --baseline baseline --output comparison.json --markdown comparison.md
```

Python API 使用相同的数据校验与评分实现：

```python
from surrogate_eval import load_reference, load_prediction, evaluate

reference = load_reference("reference.npz")
prediction = load_prediction("model.npz")
report = evaluate(reference, prediction)
print(report["aggregate"]["response_mse"])
```

从[可运行接入教程](docs/your_model.md)开始，再查阅[数据格式](docs/data_format.md)与 [API 文档](docs/api.md)。计算真实响应误差需要匹配扰动下的参考输出；普通时间序列本身不提供这些标签。

## 研究案例核验

工具已使用冻结的部分观测 Lorenz–96 研究案例进行核验，检查从原始证据、保存模型的预测重建，到通用比较报告的处理链路：

| 核验内容 | 检查依据 |
|---|---|
| 证据完整性 | 导出前对清单约束的全部 119 个文件进行 SHA-256 校验 |
| 预测重建 | 通过 NumPy 适配器重建保存模型的预测 |
| 指标一致性 | F24 历史输入回归测试以 1e-12 的绝对容差，核对导出结果与存档中的分组响应改善中位数 |
| 重复单位保留 | 同一回归测试检查导出比较仍包含八个数据集组 |

[核验记录](docs/evidence.md)分别说明历史案例检查、通用数值测试与数据约定测试。原始证据包保存在仓库外；[案例校验与导出](reproduction/README.md)需要取得该证据包。自带的两个 CPU 示例可以独立运行。

## 方法边界

软件评估的是指定有限扰动下的响应误差，不能代替未测量的完整 Jacobian 精度。局部几何分析需要调用方提供坐标一致的参考导数；逐点线性 oracle 不等于全局可学习模型。Bootstrap 区间描述固定模型和指定分组下的不确定性；物理配对、训练期尺度估计与组间独立性由研究者确认。

响应感知岭回归使用已有方法。项目的贡献在于将评估、诊断和基线比较组织为可复用的软件流程，并明确数据、聚合与来源约定。详见[指标与假设](docs/methods.md)、[基线拟合](docs/repair.md)和[相关工作](docs/related_work.md)。

## 验证与开发

Windows / Python 3.14 本地验证已通过 **38 项测试**，包含可选历史案例检查；清单约束的 **119 个证据文件**校验成功。全新 wheel 安装环境也已运行两个示例、外部模型教程及几何／噪声分析，运行时仅依赖 NumPy。这些结果验证实现行为，与跨新系统的科学有效性验证分别记录。

```sh
python -m pip install ".[dev]"
python scripts/check.py
python scripts/check.py --legacy --evidence-root PATH_TO_EVIDENCE
python -m build
python scripts/check_distribution.py
```

[GitHub Actions](https://github.com/KaR7NA16/surrogate-eval/actions/workflows/tests.yml) 在 Linux/Windows × Python 3.11/3.14 上执行测试、构建分发包并验证干净环境中的 wheel 安装。每次运行均记录对应提交的结果。开发规范见[贡献指南](CONTRIBUTING.md)，后续目标见[路线图](docs/roadmap.md)。

## 文档与引用

| 使用入口 | 深入了解 |
|---|---|
| [外部模型教程](docs/your_model.md) | [指标与假设](docs/methods.md) |
| [数据格式](docs/data_format.md) | [设计与范围](docs/design.md) |
| [Python API](docs/api.md) | [相关工作](docs/related_work.md) |
| [案例校验与导出](reproduction/README.md) | [开发与来源记录](docs/provenance.md) |

引用元数据见 [CITATION.cff](CITATION.cff)，版本记录见 [CHANGELOG.md](CHANGELOG.md)。当前 0.1.0 为 Alpha 版本，尚未发布到公共软件包索引或分配 DOI。

新软件代码、测试与文档采用 [MIT 许可](LICENSE)；历史研究材料保留单独的[许可范围](docs/licensing.md)。

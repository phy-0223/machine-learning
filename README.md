# 浙江大学机器学习课程：16 周学习工作区

这个工作区把课程路线落实为一套可以持续执行、检查和复现的学习系统。默认学习者已经学过微积分、线性代数和基础 Python，每周可投入 4–7 小时。

## 你将完成的成果

- 一次无数据泄漏的经典机器学习模型比较实验；
- 梯度下降和两层神经网络的 NumPy 从零实现；
- 一个可在 CPU 上运行的 PyTorch CNN 图像分类基线；
- 一份包含基线、改进、误差分析和伦理反思的课程项目；
- 一套公式卡、错题记录、闭卷自测与能力验收记录。

详细安排见 [16 周路线](docs/roadmap.md)，每学一个算法都填写 [六问模板](docs/algorithm-template.md)，每周在 [进度表](docs/progress.md) 打卡。

## 环境安装

建议使用 Python 3.11–3.13。PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

前 8 周只需核心环境。进入深度学习阶段后再安装 PyTorch：

```powershell
python -m pip install -e ".[deep,dev]"
```

如 PyTorch 安装方式因 CPU/GPU 平台不同，以 PyTorch 官方安装选择器给出的命令为准。

## 第一次运行

```powershell
# 传统模型比较；结果写入 outputs/classical/
python -m ml_course.classical --output outputs/classical

# NumPy 梯度下降；结果写入 outputs/gradient/
python -m ml_course.gradient_descent --output outputs/gradient

# NumPy 两层网络学习 XOR
python -m ml_course.neural_network

# 安装 deep 依赖后运行 CNN 离线基线
python -m ml_course.cnn --output outputs/cnn --epochs 8

# 自动验收
pytest -q
```

每个命令都固定随机种子并保存结构化指标或图像。`outputs/` 不纳入版本控制；提交课程项目时，应将最终指标、关键图片和配置复制到报告所引用的固定目录。

## 工作区结构

```text
docs/                 16 周路线、学习模板、公式卡和验收表
src/ml_course/        可直接运行的实验代码
tests/                自动化正确性检查
project/              大作业执行说明、实验台账和 LaTeX 报告模板
outputs/              运行后生成的指标、图表与模型（被忽略）
```

## 三条硬规则

1. **先切分，再拟合预处理。** 缺失值填补、标准化、特征选择都只在训练数据或交叉验证折内拟合。
2. **测试集只使用一次。** 选模型和调参使用交叉验证或验证集，最终方案冻结后再看测试集。
3. **报告失败与限制。** 不只展示最高分，还要说明错误样本、数据偏差、复现条件和不适用场景。

## 规范来源致谢

本工作区的经典机器学习流程参考了 Scientific Agent Skills 中的 scikit-learn 程序规范：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065


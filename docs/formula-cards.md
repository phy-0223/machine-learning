# 核心公式卡

这些卡片用于闭卷回忆，不替代完整推导。每张卡的背面应补上“假设、变量形状、何时失效”。

## 优化

- 梯度下降：$\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)$。
- 带 L2 正则：$L_{reg}=L+\lambda\lVert\theta\rVert_2^2$，梯度额外增加 $2\lambda\theta$。
- 凸函数一阶条件：$f(y)\ge f(x)+\nabla f(x)^\top(y-x)$。

## 线性与逻辑模型

- 线性模型：$\hat y=Xw+b$。
- 均方误差：$L=\frac1n\lVert Xw+b-y\rVert_2^2$。
- MSE 梯度：$\nabla_w L=\frac2nX^\top(Xw+b-y)$，$\nabla_b L=\frac2n\sum_i(\hat y_i-y_i)$。
- Sigmoid：$\sigma(z)=1/(1+e^{-z})$。
- 二分类交叉熵：$L=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$。

## 反向传播

- 链式法则：若 $L=L(a)$、$a=f(z)$、$z=Wx+b$，则 $\partial L/\partial z=(\partial L/\partial a)\odot f'(z)$。
- 参数梯度：$\partial L/\partial W=(\partial L/\partial z)x^\top$；批量形式需注意样本维所在轴。
- 数值梯度检查：$\partial f/\partial\theta_j\approx[f(\theta+\epsilon e_j)-f(\theta-\epsilon e_j)]/(2\epsilon)$。

## 分类指标

- Precision $=TP/(TP+FP)$：预测为正的样本中有多少是真的。
- Recall $=TP/(TP+FN)$：真实正样本中找回多少。
- F1 $=2PR/(P+R)$：Precision 与 Recall 的调和平均。
- Accuracy 在严重类别不平衡时可能具有误导性，应配合 PR、F1、balanced accuracy 或 ROC-AUC。

## CNN

- 单维卷积输出：$\lfloor (W-K+2P)/S\rfloor+1$。
- 卷积参数量：$C_{out}(C_{in}K_hK_w+1)$，最后的 1 是每个输出通道的偏置。

## 注意力与 GAN

- 缩放点积注意力：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$。
- GAN 原始目标：$\min_G\max_D\;\mathbb E_{x\sim p_{data}}\log D(x)+\mathbb E_z\log(1-D(G(z)))$。


# %% [markdown]
# # 前言：在写注意力之前，你需要先会这几件事
# 
# 这一节不追求“炫技”，只做一件事：
# 把后面章节会反复出现的基础组件先讲清楚，避免你在看 Self-Attention 时被训练代码分散注意力。
# 
# 你可以把这一节当成“看懂后续代码的钥匙”。
# 
# ---
# 
# ## 1. nn.Embedding 是什么，为什么在本教程里很重要
# 
# 在本教程里，我们经常把离散的整数序列当成输入，例如：
# [3, 1, 8, 2, 5]
# 
# 这些整数本身只是 ID，不带可学习的连续语义信息。
# `nn.Embedding` 的作用就是把“离散 ID”映射成“可学习向量”。
# 
# 直观理解：
# - 输入是词表索引（long 类型）
# - 输出是浮点向量
# - 每个索引都对应一行可训练参数
# 
# 张量形状通常是：
# - 输入 x: (batch, seq_len)
# - 输出 emb(x): (batch, seq_len, d_model)
# 
# 这正是后面注意力模块的标准输入形状。
# 
# ---
# 
# ## 2. nn.Linear 在这里做什么
# 
# `nn.Linear` 是最基础的线性变换层，形式上可以写成：
# y = xW + b
# 
# 在后续 Self-Attention 章节里，同一个 embedding 输出会经过三组不同的线性层，
# 分别得到 Q、K、V。你现在先理解 `Linear` 的输入输出维度就够了：
# 
# - 如果输入最后一维是 d_model
# - 线性层定义为 Linear(d_model, d_out)
# - 那么输出最后一维会变成 d_out
# 
# 例子：
# (batch, seq_len, d_model) -> (batch, seq_len, d_out)
# 
# ---
# 
# ## 3. 训练三件套：model / optimizer / criterion
# 
# 初学者经常把这三者混在一起。可以这样记：
# 
# - model：前向计算，负责“给出预测”
# - criterion：衡量预测与标签差距，得到 loss
# - optimizer：根据梯度更新参数，让下次预测更好
# 
# 只要你把这三个角色分清，任何 PyTorch 训练循环都会变得可读。
# 
# ---
# 
# ## 4. 最小训练循环逐行解释
# 
# ```python
# for i in range(epochs):
#     optimizer.zero_grad()        # 1) 清空上一轮累计梯度
#     pred = model(train_x)        # 2) 前向计算得到预测
#     loss = criterion(pred, train_y)  # 3) 计算损失
#     loss.backward()              # 4) 反向传播，计算每个参数的梯度
#     optimizer.step()             # 5) 用梯度更新参数
# ```
# 
# 为什么每轮都要 `zero_grad()`？
# 因为 PyTorch 默认会累加梯度，不清空就会把历史梯度叠加到当前轮，训练行为会失真。
# 
# 为什么 `backward()` 在 `step()` 前？
# 因为优化器更新参数需要先拿到梯度，而梯度是由 `backward()` 计算出来的。
# 
# ---
# 
# ## 5. 和后面章节的对应关系
# 
# 你在这一节理解的内容，会在后续章节直接复用：
# 
# - 第一章：Embedding + Linear 构建单头注意力，训练“找最大值下标”任务
# - 第二章：在第一章基础上扩展成多头注意力
# - 第三章：在注意力里加入 RoPE 位置编码
# - 第四章：组合成 Transformer Block（注意力 + FFN + LayerNorm + 残差）
# - 第五章：堆叠成 Encoder 并在真实数据集上训练
# 
# 换句话说，这一节看似简单，但它决定你后面是否能“读懂每一行训练代码”。
# 
# ---
# 
# ## 6. 初学者常见坑（建议先避开）
# 
# 1) 输入给 Embedding 的张量类型必须是 long。
# 2) 关注维度变化，尤其是 batch / seq_len / hidden_dim 的位置。
# 3) 先保证一小批数据能跑通，再扩大 batch 和 epoch。
# 4) loss 不下降时，先检查标签格式和输出维度是否匹配。
# 
# 如果你准备好了，就可以进入第一章，开始实现最小可用的 Self-Attention。



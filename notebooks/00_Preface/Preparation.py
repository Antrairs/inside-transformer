# %% [markdown]
# torch的基本用法
# - nn.Embedding
# - nn.Linear
# 
# 深度学习基本训练过程
# - model
# - optimizer
# - criterion
# ```python
# for i in range(epochs):
#     optimizer.zero_grad() # 清空梯度
#     pred = model(train_x) # 数据输入模型
#     loss = criterion(pred, train_y) # 计算损失函数
#     loss.backward() # 反向传播
#     optimizer.step() # 优化器更新参数
# ```



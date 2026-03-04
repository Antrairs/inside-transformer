from src.Transformer import Transformer
from src.Tokenizer import CharTokenizer
import torch
from torch import nn
import random

BATCH_SIZE = 1024
MAX_LEN = 10
VOCAB_SIZE = 14

PLUS_IDX = 10
SOS_IDX  = 11
EOS_IDX  = 12
PAD_IDX  = 13

def char_to_id(char):
    if char.isdigit(): return int(char)
    if char == '+': return PLUS_IDX
    if char == '=': return SOS_IDX
    return PAD_IDX

# ID 转字符的工具函数 (调试用)
def id_to_char(idx):
    if idx < 10: return str(idx)
    if idx == 10: return "+"
    if idx == 11: return "="
    if idx == 12: return "E"
    if idx == 13: return "P"
    return "?"

raw_vocab = "0123456789+=" 
tokenizer = CharTokenizer(raw_vocab)

print(f"词表大小: {tokenizer.vocab_size}")
print(f"映射关系: {tokenizer.char2id}")

def get_batch():
    src_batch = []
    tgt_batch = []
    label_batch = []
    
    for _ in range(BATCH_SIZE):
        # 1. 随机生成题目
        a = random.randint(0, 999) # 随机数 a
        b = random.randint(0, 999) # 随机数 b
        c = a + b                  # 答案 c
        
        # 2. 构造字符串
        # src_str: "12+34"
        src_str = f"{a}+{b}"
        # res_str: "46"
        res_str = f"{c}"
        
        # 3. 转换成 ID 列表
        # src_ids: [1, 2, 10, 3, 4]
        src_ids = [char_to_id(ch) for ch in src_str]
        
        # res_ids: [4, 6]
        res_ids = [char_to_id(ch) for ch in res_str]
        
        # 4. 填充 (Padding)
        # 为了让这一批次的数据长度整齐划一，不足 MAX_LEN 的地方要补 PAD
        
        # 处理 Src: 补 PAD 到 MAX_LEN
        # 注意：Src 不需要加 SOS/EOS，只要补齐就行
        num_pads_src = MAX_LEN - len(src_ids)
        if num_pads_src < 0: src_ids = src_ids[:MAX_LEN] # 防御性截断
        else: src_ids = src_ids + [PAD_IDX] * num_pads_src
        
        # 处理 Tgt/Label: 需要考虑 SOS 和 EOS
        # Tgt (Decoder输入): [SOS, 4, 6, PAD...]
        # Label (预测目标):  [4, 6, EOS, PAD...]
        
        tgt_seq   = [SOS_IDX] + res_ids
        label_seq = res_ids + [EOS_IDX]
        
        # 补齐 Tgt
        num_pads_tgt = MAX_LEN - len(tgt_seq)
        if num_pads_tgt < 0: tgt_seq = tgt_seq[:MAX_LEN]
        else: tgt_seq = tgt_seq + [PAD_IDX] * num_pads_tgt
            
        # 补齐 Label
        num_pads_label = MAX_LEN - len(label_seq)
        if num_pads_label < 0: label_seq = label_seq[:MAX_LEN]
        else: label_seq = label_seq + [PAD_IDX] * num_pads_label

        # 5. 加入 Batch
        src_batch.append(src_ids)
        tgt_batch.append(tgt_seq)
        label_batch.append(label_seq)
    
    # 6. 转为 Tensor
    # 形状都是 [BATCH_SIZE, MAX_LEN]
    src = torch.tensor(src_batch).long()
    tgt = torch.tensor(tgt_batch).long()
    label = torch.tensor(label_batch).long()
    
    return src, tgt, label

device = torch.device("cuda")
model = Transformer(input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE,
                    d_model=128, d_ff=512,
                    num_heads=8, num_layers=8).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=50, 
)

# def get_batch():
#     # 动态生成随机数据
#     x = torch.randint(0, 10, (BATCH_SIZE, 10)).long()
#     y, _ = torch.sort(x, dim=1)
    
#     # 构造 Decoder 输入
#     sos = torch.ones(BATCH_SIZE, 1).long() * SOS_TOKEN
#     tgt = torch.cat([sos, y[:, :-1]], dim=1)
#     return x, y, tgt

def train(epoch_num, save_path):
    for epoch in range(epoch_num):
        src, tgt, label = get_batch()
        src, tgt, label = src.to(device), tgt.to(device), label.to(device)
        
        optimizer.zero_grad()

        output = model(src, tgt)
        loss = criterion(output.reshape(-1, VOCAB_SIZE), label.reshape(-1))
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epoch_num}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), save_path)

def predict_addition(question_str):
    """
    输入: "12+34"
    输出: "46"
    """
    model.eval() # 开启评估模式 (关闭 Dropout)
    
    # --- 1. 数据预处理 (把字符串变成 Tensor) ---
    # 比如 "12+34" -> [1, 2, 10, 3, 4]
    src_ids = [char_to_id(c) for c in question_str]
    
    # 补齐长度 (虽然 Batch=1 不补也没事，但为了和训练一致最好补)
    if len(src_ids) < MAX_LEN:
        src_ids = src_ids + [PAD_IDX] * (MAX_LEN - len(src_ids))
    
    # 增加 Batch 维度: [Len] -> [1, Len]
    src_tensor = torch.tensor(src_ids).long().unsqueeze(0).to(device)
    
    # --- 2. 自回归生成 (Autoregressive Generation) ---
    # 初始 Decoder 输入只有 SOS ("=")
    # [1, 1]
    curr_tgt = torch.tensor([[SOS_IDX]]).long().to(device)
    
    result_ids = []
    
    with torch.no_grad():
        # 最多生成 MAX_LEN 次，防止死循环
        for _ in range(MAX_LEN):
            # 这里的 src_tensor 是完整的题目
            # curr_tgt 是目前已经算出来的结果
            output = model(src_tensor, curr_tgt)
            
            # 取最后一个时间步的预测结果
            # [1, Seq, Vocab] -> [1, Vocab]
            next_token_logits = output[:, -1, :]
            
            # 选概率最大的 ID
            next_token_id = next_token_logits.argmax(dim=-1).item()
            
            # 如果预测到了 EOS (结束符) 或 PAD，就停止
            if next_token_id == EOS_IDX or next_token_id == PAD_IDX:
                break
                
            # 记录结果
            result_ids.append(next_token_id)
            
            # 把这个新字拼接到输入里，准备下一轮预测
            # [1, Len] -> [1, Len+1]
            next_token_tensor = torch.tensor([[next_token_id]]).long().to(device)
            curr_tgt = torch.cat([curr_tgt, next_token_tensor], dim=1)
            
    # --- 3. 解码 (ID -> 字符串) ---
    # [4, 6] -> "46"
    res_str = "".join([id_to_char(idx) for idx in result_ids])
    return res_str


if __name__ == "__main__":
    # 1. 开始训练
    # 加法比排序难，建议多练一会儿，比如 2000-3000 轮
    # 观察 Loss，如果降到 0.0x 级别才算学会
    print("🚀 正在教 AI 做数学题...")
    train(epoch_num=1000, save_path="checkpoints/addition_transformer.pth") 
    
    # 2. 验证测试
    print("\n✨ 最终考试时间 ✨")
    
    test_cases = [
        "1+1",
        "99+1",    # 进位测试
        "500+500", # 进位测试
        "123+456",
        "0+0",     # 零测试
        "7+8",     # 小数字测试
        "999+999", # 边界测试
        "250+375",
        "1234+5678", # 超出长度测试
    ]
    
    for q in test_cases:
        pred = predict_addition(q)
        
        # 用 Python 算个真值来对比
        parts = q.split('+')
        real_ans = str(int(parts[0]) + int(parts[1]))
        
        flag = "✅" if pred == real_ans else "❌"
        print(f"题目: {q:<10} | AI回答: {pred:<5} | 正确答案: {real_ans:<5} | {flag}")
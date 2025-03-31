import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, trainers  # 新增BPE依赖
import gc
import tiktoken

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

## 超参设置
batch_size = 16 # 批处理大小，并行训练多少个独立序列
block_size = 32 # 推理的最大上下文长度
max_iters = 10000 # 最大训练迭代次数
eval_interval = 100 # 多少次迭代评估一次
learning_rate = 1e-3 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'# 是否使用GPU
eval_iters = 200 # 评估模型时的迭代次数
n_embd = 256 # embedding后的向量维度
n_head = 4 # 多头注意力的头数
n_layer = 8 # transformer层数
dropout = 0.0 # dropout率
torch.manual_seed(42) # 随机种子

from bpe_tokenizer import BPETokenizer

# 替换原有字符处理部分
# --------------------------------------------------
# 初始化BPE分词器
bpe_tokenizer = BPETokenizer.load("bpe_tokenizer.json")  # 加载训练好的模型
#gpt2_tokenizer

# 新的编解码函数
encode = lambda s: bpe_tokenizer.encode(s)
decode = lambda ids: bpe_tokenizer.decode(ids)

# 处理输入数据
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 编码整个文本
encoded_ids = encode(text)
data = torch.tensor(encoded_ids, dtype=torch.long)

# 更新词汇量参数
vocab_size = bpe_tokenizer.vocab_size

## 数据集划分以及数据分批
# torch.long是长整形
data = torch.tensor(encode(text), dtype=torch.long)
# 90%划分训练集和验证集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # 生成一个batch的数据，x为输入，y是target
    data = train_data if split == 'train' else val_data
    #data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # y相对于x移位一个token，因为训练的目标就是预测下一个token
    x, y = x.to(device), y.to(device)
    return x, y

## 模型评估
# 评估模型在训练集和验证集上的平均损失，评估eval_iters次
@torch.no_grad() # 仅评估
def estimate_loss():
    out = {}
    model.eval() # 模型评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 模型训练模式
    return out

## 模型定义
# 多头注意力模型中的单个头
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        # QKV映射矩阵
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 创建一个下三角矩阵，并将其注册为模型的一个缓冲区
        # 用于计算masked self-attention时作为mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # batch_size, block_size, embed_dim
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        # 计算attention分数，1/sqrt(d_k) -> C**-0.5
        #wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # 修正为
        head_size = q.size(-1)
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5
        # 利用下三角矩阵对attention进行mask填充，保证当前的token只能看到它之前的token，看不到后面的
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # softmax归一化权重
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 得到的权重再和value相乘
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
# 多头注意力
# 实现多头自注意力机制，并行计算自注意力，然后将结果拼接起来
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 输出映射线性层
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # 对每个注意力头h进行计算，然后将结果在最后一个维度上拼接起来，得到out
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
# 模型FeedFoward
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        # 两个线性层和一个ReLU激活函数，以及一个dropout层
        # 中间升维 embed_dim->4*embed_dim，然后降维回去
        # self.net = nn.Sequential(
        #     nn.Linear(n_embd, 4 * n_embd),
        #     nn.ReLU(),
        #     nn.Linear(4 * n_embd, n_embd),
        #     nn.Dropout(dropout),
        # )
        self.layer1 = nn.Linear(n_embd, 4 * n_embd)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x = self.net(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
# 模型Block定义
# 实现一个Transformer Block
# 包含layerNorm, multiheadattention, layerNorm, feedforward
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 每一个头的size
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    # Transformer结构
    # pre-norm->self-attention->残差链接->pre-norm->FFN->残差链接
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 定义一个二元语言模型BigramLanguageModel
# 训练一个二元语言模型，并生成新的文本
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # embedding表和位置嵌入表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 使用BPE词汇量
        #self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 堆叠多个Transformer Block，总共有n_layer个Transformer Block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 最后一个layerNorm
        # 输出线性层，将嵌入向量映射到词汇表大小的向量
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # batch_size, block_size
        B, T = idx.shape
        # 首先，从词嵌入表和位置嵌入表中获取嵌入，将它们相加得到x
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # 然后，x输入到Transformer Block中，得到输出x
        x = self.blocks(x) # (B,T,C)
        # 最后，将输出x输入到输出线性层中，得到logits
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, targets)
            # 返回每个token的对数概率
            probs = F.softmax(logits, dim=-1)  # (B*T, vocab_size)
            log_probs = torch.log(probs)  # (B*T, vocab_size)

        return logits, loss
        #return log_probs, loss

    def generate(self, idx, max_new_tokens):
        # 生成新的文本，从当前的索引idx开始，每次生成一个新的token，直到生成max_new_tokens个token为止
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 将当前的索引裁剪到最后的block_size个token
            idx_cond = idx[:, -block_size:]
            # 获取预测的logits，只关注最后一个时间步
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            # 将logits应用softmax得到概率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从分布中采样一个next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将采样的索引添加到当前的序列中
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

## 模型训练
# 实例化
print(f"Before CUDA operation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
torch.zeros(1, device='cuda')  # 触发 CUDA 初始化
print(f"After CUDA operation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
model = BigramLanguageModel()
m = model.to(device) # 移动到GPU上
# 打印模型参数数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# 检查模型参数的数据类型
for name, param in model.named_parameters():
    print(f"Parameter: {name}, DataType: {param.dtype}")
# 训练模型
# 创建一个AdamW优化器，学习率为learning_rate，将模型的参数传递给优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 添加梯度裁剪
#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用学习率热身
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))
# = torch.optim.lr_scheduler.LambdaLR(
  #  optimizer, lambda step: min(step/1000, 1))  # 前1000步热身


# 初始化列表来存储损失值
train_losses = []
val_losses = []

for iter in range(max_iters):
    # 每过100次迭代，就评估模型的损失，打印训练损失和验证损失
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        #perplexity = calculate_perplexity(model, val_loader)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val perplexity {perplexity:.2f}")

        # 将损失值添加到列表中
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
    # 采样一个batch的数据
    xb, yb = get_batch('train')
    # 计算损失
    logits, loss = model(xb, yb)
    # 将梯度清零，PyTorch默认会累积梯度，所以需要手动清零
    optimizer.zero_grad(set_to_none=True)
    # 计算梯度
    loss.backward()
    # 更新优化器的参数
    optimizer.step()

# 绘制损失曲线图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# 保存图片
plt.savefig('loss_plot.png')

# 生成新的文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
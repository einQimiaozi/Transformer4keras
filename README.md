# Transformer4keras
基于keras实现的transformer
# 使用方法
  - 使用keras中的Input作为输入连接到Transformer
  - 根据下游任务可以在Transformer后接不同的net，但是不建议接太复杂
  - 也可以直接使用MultiHeadAttention后接下游任务
# api
  - Embedding(vocab_size, model_dim)(layers) 将上一层输入的单词token做embedding,type=layers,参数:vocab_size=单词数量|type=int model_dim=词嵌入维度|type=int
  
  - PositionalEncoding(model_dim)(layers) 对上一层做位置编码,type=layers,参数:model_dim=词嵌入维度|type=int
  
  - Add()[layers1,layers2]) 将layers1和layers2相加,type=layers,参数：None
  
  - MultiHeadAttention(heads,model_dim)([layers,layers,layers]) 对layers做多头注意力处理,type=layers,参数:heads=多头数量|type=int model_dim=词嵌入维度|type=int
  
  - Transformer(num_layers=num_layers,vocab_size=vocab_size,heads=8,model_dim=model_dim,drop_rate=0.2,units_dim=512,learning_rate=1e-4)(layers)
    
    - 在layers后接整个transformer架构,type=layers 
      
    - 参数:
        
      - num_layers=transformer中编码解码器的数量
      
# 案例
  - 1.

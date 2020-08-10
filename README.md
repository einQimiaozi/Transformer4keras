# Transformer4keras
  
基于keras实现的transformer
  
# 使用方法
  
  - 使用keras中的Input作为输入连接到Transformer
    
  - 根据下游任务可以在Transformer后接不同的net，但是不建议接太复杂
    
  - 也可以直接使用MultiHeadAttention后接下游任务
# api
  
  - Embedding(vocab_size, model_dim=512)(layers) 将上一层输入的单词token做embedding,type=layers,参数:vocab_size=单词数量|type=int model_dim=词嵌入维度|type=int
  
  - PositionalEncoding(model_dim=512)(layers) 对上一层做位置编码,type=layers,参数:model_dim=词嵌入维度|type=int
  
  - Add()[layers1,layers2]) 将layers1和layers2相加,type=layers,参数：None
  
  - MultiHeadAttention(heads=8,model_dim=512)([layers,layers,layers]) 对layers做多头注意力处理,type=layers,参数:heads=多头数量|type=int model_dim=词嵌入维度|type=int
  
  - Transformer(num_layers=num_layers,vocab_size=vocab_size,heads=8,model_dim=512,drop_rate=0.2,units_dim=512,epsilon=0.001)(layers)
    
    - 在layers后接整个transformer架构,type=layers 
      
    - 参数:
        
      - num_layers=transformer中编码解码器的数量|type=int
        
      - vocab_size=单词数量|type=int
        
      - heads=多头数量|type=int
        
      - model_dim=词嵌入维度|type=int
        
      - drop_rate=encoder和decoder中各种连接处的dropout参数|type=float
        
      - units_dim=encoder和decoder中全连接网络部分的神经元数|type=int
        
      - epsilon=encoder和decoder中LayerNormalization层的参数epsilon
  
# tips
  
  - 训练参数较多，强烈建议在程序开头设置
    
  '''python
  import os
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  '''
    
  - 任何显卡第一次都不要轻易将下游任务的batch_size设置超过32，一般从32开始试，如果训练比较轻松可以加大，显存爆炸则降低
    
  - 由于Transformer4keras基于keras，所以请提前安装并配置好keras
  
# 案例
    
  - 0.首先导入必要模块
  ```python
  from tensorflow.keras.datasets import imdb
  from tensorflow.keras.preprocessing import sequence
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.layers import Input, Dense,GlobalAveragePooling1D,Dropout
  from tensorflow.keras.models import Model
  from tensorflow.keras.optimizers import Adam,RMSprop
  from Transformer import Transformer,MultiHeadAttention,Embedding,PositionalEncoding,Add
  ```
  - 1.使用MultiHeadAttention训练imdb数据
  ```python
  vocab_size = 5000
  maxlen = 256
  model_dim = 512     # 词嵌入的维度
  batch_size = 32
  epochs = 10
  num_layers = 2

  # 读取imdb数据
  print("Data downloading and pre-processing ... ")
  (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=maxlen, num_words=vocab_size)
  x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
  x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  inputs = Input(shape=(maxlen,), name="inputs")
  embeddings = Embedding(vocab_size=vocab_size, model_dim=model_dim)(inputs)
  encodings = PositionalEncoding(model_dim)(embeddings)
  encodings = Add()([embeddings, encodings])
  x = MultiHeadAttention(heads=8,model_dim=model_dim)([encodings, encodings, encodings])
  x = GlobalAveragePooling1D()(x)
  x = Dropout(0.2)(x)
  x = Dense(10, activation='relu')(x)
  outputs = Dense(2, activation='softmax')(x)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                loss='categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())
  print("Model Training ... ")
  model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs, validation_split=0.2)
  ```
    
  - 2.使用Transformer训练imdb数据
    
  ```python
  vocab_size = 5000
  maxlen = 256
  model_dim = 512     # 词嵌入的维度
  batch_size = 32
  epochs = 10
  num_layers = 2

  print("Data downloading and pre-processing ... ")
  (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=maxlen, num_words=vocab_size)
  x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
  x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  
  print('Model building ... ')
  inputs = Input(shape=(maxlen,),name="inputs")
  transformer = Transformer(num_layers=num_layers,vocab_size=vocab_size,heads=8,model_dim=model_dim,
                            drop_rate=0.2,units_dim=512,epsilon=0.001)(inputs)
  outputs = Dense(2, activation='softmax')(transformer)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=RMSprop(learning_rate=4e-4),
                loss='categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())
  print("Model Training ... ")
  model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs, validation_split=0.2)
  ```
  

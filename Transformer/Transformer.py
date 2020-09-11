from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,LayerNormalization
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D

'''
牢记，Transformer架构在处理文本的时候是以词为单位
所以代码中的所有embedding都代表的是一句话，而不是输入的整个语料库
计算全部使用32位
根据keras官方文档,build方法定义权重,call方法编写逻辑,compute_output_shape方法改变输入->输出张量的形状变化
关于mask我要多说两句，原本我以为mask是直接形成下三角矩阵的，仔细看论文才发现，encoder中的mask其实就是padding，没有那么复杂，decoder里的mask才是下三角......
'''

# 词嵌入为3d张量
class Embedding(Layer):
    def __init__(self,model_dim,vocab_size,**kwargs):
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        super(Embedding,self).__init__(**kwargs)
    # 采用随机初始化方法初始化一个和输入大小一样的embedding
    def build(self,input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.model_dim),
            initializer='glorot_uniform',
            name="embeddings")
        super(Embedding, self).build(input_shape)
    # 将输入的token转换成embedding，同时做scale
    def call(self,token,scale=0.5):
        # 转换类型
        if K.dtype(token) != "int32":
            token = K.cast(token,"int32")
        # 按token取embedding对应行
        embedding = K.gather(self.embeddings,token)
        embedding = embedding*(self.model_dim**scale)
        return embedding
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'vocab_size' : self.vocab_size
        })
    def compute_output_shape(self, input_shape):
        return input_shape + (self._model_dim,)

# 接embedding层，位置编码
class PositionalEncoding(Layer):
    def __init__(self,model_dim,**kwargs):
        self.model_dim = model_dim
        super(PositionalEncoding, self).__init__(**kwargs)
    def get_angles(self,pos,i,d_model):
        return pos/(np.power(10000, (2 * (i//2)) / np.float32(d_model)))
    def call(self,embedding):
        # 输入的是embedding，所以embedding的行就是当前这句话
        # embedding.shape[0]=数据量
        # embedding.shape[1]=句子长度
        # embedding.shape[2]=词嵌入维度
        sentence_length=embedding.shape[1]
        positional_encoding = np.zeros(shape=(sentence_length,self.model_dim))
        # 计算sin/cos位置编码(论文里有公式，懒得备注了)
        for pos in range(sentence_length):
            for i in range(self.model_dim):
                positional_encoding[pos, i] = self.get_angles(pos,i,self.model_dim)
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # 用于偶数索引2i
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # 用于奇数索引2i+1
        return K.cast(positional_encoding, 'float32')
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mmodel_dim': self.model_dim,
        })
    def compute_output_shape(self,input_shape):
        return input_shape

# 将embedding和positional encoding相加
class Add(Layer):
    def __init__(self,**kwargs):
        super(Add, self).__init__(**kwargs)
    # 这里的inputs指embedding+positional encoding
    def call(self, inputs):
        input_a, input_b = inputs
        res = input_a+input_b
        return res
    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ScaledDotProductAttention(Layer):
    def __init__(self,mode,**kwargs):
        assert mode == "encoder" or mode == "decoder", "The parameter 'mode' can only receive two values, 'encoder' and 'decoder'."
        self.masking_num = -2**32
        self.mode = mode
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    # padding mask
    # 将0值位置置为一个极小的负数，使得softmax时该值接近0
    def padding_mask(self, QK):
        padding = tf.cast(tf.equal(QK,0),tf.float32)
        padding *= self.masking_num
        return QK+padding
    # sequence mask(传说中的下三角)
    def sequence_mask(self,QK):
        # 初始化下三角矩阵
        seq_mask = 1-tf.linalg.band_part(tf.ones_like(QK), -1, 0)
        seq_mask *= self.masking_num
        return QK+seq_mask
    # 输入为qkv三个矩阵和一个mask矩阵
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        queries, keys, values = inputs
        # 转换为32位
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
        # Qk计算
        matmul = tf.matmul(queries,keys,transpose_b=True)
        dk = tf.cast(tf.shape(keys)[-1],tf.float32)
        matmul = matmul / tf.sqrt(dk) # QxK后缩放dk**(0.5)
        # mask层,区别encoder和decoder部分
        if self.mode == "encoder":
            matmul = self.padding_mask(matmul)
        else:
            matmul = self.sequence_mask(matmul)
        softmax_out = K.softmax(matmul)  # SoftMax层
        return K.batch_dot(softmax_out, values) # 最后乘V
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'masking_num': self.masking_num,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

class MultiHeadAttention(Layer):
    def __init__(self, heads=8,model_dim=512,mode="encoder",trainable=True,**kwargs):
        self.heads = heads
        self.head_dim = model_dim//heads
        self.mode = mode
        self.trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)
    # 随机初始化Q K V矩阵权重
    def build(self,input_shape):
        self.weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_queries')
        self.weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_keys')
        self.weights_values = self.add_weight(
            shape=(input_shape[2][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_values')
        self.shape= input_shape
        super(MultiHeadAttention, self).build(input_shape)
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        # 注意，这里传入的qkv并不是真正的qkv，而是上一层的embedding(3个),之后乘权重才是真正的qkv
        queries, keys, values = inputs
        # 初始化
        queries_linear = K.dot(queries, self.weights_queries)
        keys_linear = K.dot(keys, self.weights_keys)
        values_linear = K.dot(values, self.weights_values)
        # 多头切割
        queries_multi_heads = tf.concat(tf.split(queries_linear, self.heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self.heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self.heads, axis=2), axis=0)

        att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
        attention = ScaledDotProductAttention(mode=self.mode)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self.heads, axis=0), axis=2)
        return outputs
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'head_dim': self.head_dim,
            'heads': self.heads,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

# encoder和decoder都要用到的前向传播
def FeedForwardNetwork(units_dim,model_dim):
    return Sequential([Dense(units_dim, activation='relu'),Dense(model_dim)])

class EncoderLayer(Layer):
    def __init__(self,heads=8,model_dim=512,units_dim=512,epsilon=0.001,drop_rate=0.2,**kwargs):
        self.heads = heads
        self.model_dim = model_dim
        self.multi_head_attention = MultiHeadAttention(self.heads,model_dim=model_dim,mode="encoder")
        self.ff_netword = FeedForwardNetwork(units_dim,model_dim)
        self.layer_norm1 = LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = LayerNormalization(epsilon=epsilon)
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)
        self.dropout3 = Dropout(drop_rate)
        super(EncoderLayer, self).__init__(**kwargs)
    # traning是个bool
    def call(self,encodings,training=True):
        attn_output = self.multi_head_attention([encodings,encodings,encodings])
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.layer_norm1(encodings + attn_output)

        ffn_output = self.ff_netword(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        out3 = self.dropout3(out2)
        return out3
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

class DecoderLayer(Layer):
    def __init__(self,heads=8,model_dim=512,units_dim=512,epsilon=0.001,drop_rate=0.2,**kwargs):
        self.heads = heads
        self.model_dim = model_dim
        # decoder中使用sequence mask和padding mask相加，所以有两个multi_head_attention
        self.multi_head_attention1 = MultiHeadAttention(self.heads,model_dim=self.model_dim,mode="decoder")
        self.multi_head_attention2 = MultiHeadAttention(self.heads,model_dim=self.model_dim,mode="encoder")
        self.ff_netword = FeedForwardNetwork(units_dim,model_dim)
        self.layer_norm1 = LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = LayerNormalization(epsilon=epsilon)
        self.layer_norm3 = LayerNormalization(epsilon=epsilon)
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)
        self.dropout3 = Dropout(drop_rate)
        self.dropout4 = Dropout(drop_rate)
        super(DecoderLayer,self).__init__(**kwargs)
    def call(self,encodings,encoder_outpus,training=True):
        attn_output1 = self.multi_head_attention1([encodings,encodings,encodings])
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layer_norm1(encodings + attn_output1)

        attn_output2 = self.multi_head_attention2([encoder_outpus,encoder_outpus,out1])
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layer_norm2(out1 + attn_output2)

        ffn_output = self.ff_netword(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(out2 + ffn_output)
        out4 = self.dropout4(out3)

        return out4
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

class Encoder(Layer):
    def __init__(self,num_layers,vocab_size,heads=8,model_dim=512,drop_rate=0.2,units_dim=512,epsilon=0.001,**kwargs):
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = Dropout(drop_rate)
        self.embeddings_layer = Embedding(model_dim=self.model_dim,vocab_size=vocab_size)
        self.encodings_layer = PositionalEncoding(model_dim=self.model_dim)
        self.enc_layers = [EncoderLayer(model_dim=model_dim,heads=heads,units_dim=units_dim,drop_rate=drop_rate,epsilon=epsilon)
                           for _ in range(num_layers)]
        super(Encoder,self).__init__(**kwargs)
    def call(self,inputs,training=True):
        embeddings = self.embeddings_layer(inputs)
        encodings = self.encodings_layer(embeddings)
        encodings = Add()([embeddings,encodings])
        outputs = self.dropout(encodings,training=training)
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs,training=training)
        return outputs
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config

class Decoder(Layer):
    def __init__(self,num_layers,vocab_size,heads=8,model_dim=512,drop_rate=0.2,units_dim=512,epsilon=0.001,**kwargs):
        self.model_dim = model_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = Dropout(drop_rate)
        self.embeddings_layer = Embedding(self.model_dim,vocab_size)
        self.encodings_layer = PositionalEncoding(self.model_dim)
        self.dec_layers = [DecoderLayer(model_dim=model_dim,heads=heads,units_dim=units_dim,drop_rate=drop_rate,epsilon=epsilon)
                       for _ in range(num_layers)]
        super(Decoder, self).__init__(**kwargs)
    def call(self,inputs,encoder_outputs,training=True):
        embeddings = self.embeddings_layer(inputs)
        encodings = self.encodings_layer(embeddings)
        encodings = Add()([embeddings,encodings])
        outputs = self.dropout(encodings,training=training)
        for i in range(self.num_layers):
            outputs = self.dec_layers[i](outputs,encoder_outputs,training=training)
        return outputs
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'model_dim': self.model_dim,
            'heads': self.heads,
        })
        return config

class Transformer(Layer):
    def __init__(self,num_layers,vocab_size,heads=8,model_dim=512,drop_rate=0.2,units_dim=512,epsilon=0.001,**kwargs):
        self.encoder = Encoder(num_layers=num_layers,vocab_size=vocab_size,heads=heads,model_dim=model_dim,
                               drop_rate=drop_rate,units_dim=units_dim,epsilon=epsilon)
        self.decoder = Decoder(num_layers=num_layers,vocab_size=vocab_size,heads=heads,model_dim=model_dim,
                               drop_rate=drop_rate,units_dim=units_dim,epsilon=epsilon)
        self.output_layer = GlobalAveragePooling1D()
        super(Transformer,self).__init__(**kwargs)
    def call(self,inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(inputs,encoder_outputs)
        outputs = self.output_layer(decoder_outputs)
        return outputs
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
        })
        return config
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],self._vocab_size)

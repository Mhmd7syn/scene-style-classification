import tensorflow as tf
from tensorflow.keras import layers, models

# ================= COMPLETE CUSTOM CLIP CLASSES =================
@tf.keras.utils.register_keras_serializable()
class QuickGELU(layers.Layer):
    def call(self, x):
        return x * tf.sigmoid(1.702 * x)

@tf.keras.utils.register_keras_serializable()
class CLIPAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = layers.Dense(embed_dim, use_bias=True)
        self.k_proj = layers.Dense(embed_dim, use_bias=True)
        self.v_proj = layers.Dense(embed_dim, use_bias=True)
        self.out_proj = layers.Dense(embed_dim, use_bias=True)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = tf.reshape(q, [B, N, self.num_heads, self.head_dim])
        k = tf.reshape(k, [B, N, self.num_heads, self.head_dim])
        v = tf.reshape(v, [B, N, self.num_heads, self.head_dim])

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        attn_weights = tf.matmul(q, k, transpose_b=True) * self.scale
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [B, N, C])
        
        return self.out_proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class CLIPMLP(layers.Layer):
    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc1 = layers.Dense(intermediate_size)
        self.activation = QuickGELU()
        self.fc2 = layers.Dense(hidden_size)
        
    def call(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class CLIPEncoderLayer(layers.Layer):
    def __init__(self, hidden_size, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        
    def call(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x
        
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
        })
        return config

def get_clip_transformer_class(default_hidden, default_patch, default_heads, default_layers, default_inter):
    
    @tf.keras.utils.register_keras_serializable()
    class CLIPVisionTransformer(models.Model):
        def __init__(self, 
                     image_size=224,
                     patch_size=default_patch,
                     hidden_size=default_hidden,
                     num_layers=default_layers,
                     num_heads=default_heads,
                     intermediate_size=default_inter,
                     projection_dim=768, # Projection dim often kept flexible
                     **kwargs):
            super().__init__(**kwargs)
            self.image_size = image_size
            self.patch_size = patch_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.intermediate_size = intermediate_size
            self.projection_dim = projection_dim
            
            self.num_patches = (image_size // patch_size) ** 2
            
            self.patch_embedding = layers.Conv2D(
                hidden_size, kernel_size=patch_size, strides=patch_size,
                padding='valid', use_bias=False, name="patch_embedding"
            )
            
            self.class_embedding = self.add_weight(
                name="class_embedding", shape=(hidden_size,),
                initializer="zeros", trainable=False
            )
            
            self.positional_embedding = self.add_weight(
                name="positional_embedding", shape=(self.num_patches + 1, hidden_size),
                initializer="zeros", trainable=False
            )
            
            self.ln_pre = layers.LayerNormalization(epsilon=1e-5)
            self.transformer_layers = [
                CLIPEncoderLayer(hidden_size, num_heads, intermediate_size, name=f"encoder_layer_{i}")
                for i in range(num_layers)
            ]
            self.ln_post = layers.LayerNormalization(epsilon=1e-5)
            self.projection = layers.Dense(projection_dim, use_bias=False, name="visual_projection")
            
        def build(self, input_shape):
            super().build(input_shape)
            self.built = True
        
        def call(self, inputs):
            B = tf.shape(inputs)[0]
            x = self.patch_embedding(inputs)
            x = tf.reshape(x, [B, self.num_patches, self.hidden_size])
            class_tokens = tf.tile(tf.reshape(self.class_embedding, [1, 1, -1]), [B, 1, 1])
            x = tf.concat([class_tokens, x], axis=1)
            x = x + self.positional_embedding
            x = self.ln_pre(x)
            for layer in self.transformer_layers: x = layer(x)
            x = self.ln_post(x)
            class_token = x[:, 0, :]
            return class_token

        def get_config(self):
            config = super().get_config()
            config.update({
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "projection_dim": self.projection_dim,
            })
            return config
            
    return CLIPVisionTransformer
# ================= END CUSTOM CLIP CLASSES =================
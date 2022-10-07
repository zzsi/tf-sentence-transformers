
import tensorflow as tf
from tensorflow import keras
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TFAutoModel


class SentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, tokenizer, model, embedding_dim=384, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_dim = embedding_dim
    
    @classmethod
    def from_pretrained(cls, model_name: str="sentence-transformers/all-MiniLM-L6-v2", from_pt=True):
        # TODO: get embedding_dim from the model.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pretrained_setr_tf = TFAutoModel.from_pretrained(model_name, from_pt=from_pt)
        embedding_dim = 384
        return cls(tokenizer=tokenizer, model=model, embedding_dim=embedding_dim)

        
    def tf_encode(self, inputs):
        def encode(inputs):
            inputs = [x[0].decode("utf-8") for x in inputs.numpy()]
            outputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='tf')
            return outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask']
        return tf.py_function(func=encode, inp=[inputs], Tout=[tf.int32, tf.int32, tf.int32])
    
    def process(self, i, t, a):
      def __call(i, t, a):
        model_output = self.model({'input_ids': i.numpy(), 'token_type_ids': t.numpy(), 'attention_mask': a.numpy()})
        return model_output[0]
      return tf.py_function(func=__call, inp=[i, t, a], Tout=[tf.float32])

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = tf.squeeze(tf.stack(model_output), axis=0)
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        a = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        b = tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
        embeddings = a / b
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings
      
    def call(self, inputs):
        input_ids, token_type_ids, attention_mask = self.tf_encode(inputs)
        model_output = self.process(input_ids, token_type_ids, attention_mask)
        embeddings = self.mean_pooling(model_output, attention_mask)
        embeddings = tf.reshape(embeddings, (-1, self.embedding_dim))
        return embeddings
    
    def test_call(self):
        inputs = ['This is a test sentence.']
        embedding = self(inputs)
        print(embedding.shape)


# If using sentence-transformers (torch):
# setr = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# embedding = setr.encode(sentence)

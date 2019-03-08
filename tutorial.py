# This is a tutorial on using this library
# first off we need a text_encoder so we would know our vocab_size (and later on use it to encode sentences)
import os
import sys
from data.vocab import SentencePieceTextEncoder  # you could also import OpenAITextEncoder

sentence_piece_encoder = SentencePieceTextEncoder(text_corpus_address='openai/model/params_shapes.json',
                                                  model_name='tutorial', vocab_size=20)

# now we need a sequence encoder
from transformer.model import create_transformer

sequence_encoder_config = {
    'embedding_dim': 6,
    'vocab_size': sentence_piece_encoder.vocab_size,
    'max_len': 8,
    'trainable_pos_embedding': False,
    'num_heads': 2,
    'num_layers': 3,
    'd_hid': 12,
    'use_attn_mask': True
}
sequence_encoder = create_transformer(**sequence_encoder_config)
import keras

assert type(sequence_encoder) == keras.Model

# now look at the inputs:
print(sequence_encoder.inputs)  # tokens, segment_ids, pos_ids, attn_mask

# for outputs we have:
print(sequence_encoder.outputs)

from data.dataset import TaskMetadata, TaskWeightScheduler

tasks = [TaskMetadata('lm', is_token_level=True,
                      num_classes=sentence_piece_encoder.vocab_size + sentence_piece_encoder.SPECIAL_COUNT,
                      dropout=0,
                      weight_scheduler=TaskWeightScheduler(active_in_pretrain=True, active_in_finetune=False,
                                                           pretrain_value=1.0))]

class ComplexTaskWeightScheduler(TaskWeightScheduler):  # note: this is an example, it is not a clean code
    def __init__(self, number_of_pretrain_steps, number_of_finetune_steps):
        super().__init__(active_in_pretrain=True, active_in_finetune=True)
        self.number_of_pretrain_steps = number_of_pretrain_steps
        self.number_of_finetune_steps = number_of_finetune_steps

    def get(self, is_pretrain: bool, step: int) -> float:
        return step / (self.number_of_pretrain_steps if is_pretrain else self.number_of_finetune_steps)


number_of_pretrain_steps = 100
number_of_finetune_steps = 100
# in this task we are going to count the number of tokens in a sentence and predict if it's odd or not
tasks.append(TaskMetadata('odd', is_token_level=False, num_classes=2, dropout=0.3,
                          weight_scheduler=ComplexTaskWeightScheduler(number_of_pretrain_steps,
                                                                      number_of_finetune_steps)))

# and let's add a unsolvable task for fun
tasks.append(TaskMetadata('lm_random', is_token_level=True,
                          num_classes=sentence_piece_encoder.vocab_size + sentence_piece_encoder.SPECIAL_COUNT,
                          dropout=0.3,
                          weight_scheduler=TaskWeightScheduler(active_in_pretrain=True, active_in_finetune=True,
                                                               pretrain_value=0.5)))

from data.dataset import Sentence, TokenTaskData, SentenceTaskData, TextEncoder
from data.lm_dataset import _create_batch
import random


def tutorial_batch_generator(vocab_size: int, max_len: int, batch_size: int, steps: int):
    def sentence_generator():
        for _ in range(steps):
            # for each sentence we are going to generate up to max_len tokens
            seq_len = random.randint(1, max_len - 1)
            # and this is their ids (in reality we have to use our TextEncoder instance here)
            tokens = [random.randrange(vocab_size) for _ in range(seq_len)]
            # we manually set the last token to EOS (which we will see how it's calculated)
            tokens[-1] = eos_id
            yield Sentence(
                tokens=tokens,
                padding_mask=[True] * seq_len,  # it means that non of the original tokens are padding
                segments=[0] * seq_len,  # for this simple example we are going to use segment_a(0) for all of them
                token_classification={  # we put labels here (for token level tasks)
                    # name_of_the_task: TokenTaskData(target(aka label), label_mask)
                    # there might be situations that you are only interested in predictions for certain tokens,
                    # you can use mask in those situations (see the bert paper to understand this)
                    'lm': TokenTaskData(tokens, [True] * seq_len),
                    # this task is unsolvable so we will see the loss not decreasing
                    'lm_random': TokenTaskData([random.randrange(vocab_size) for i in range(seq_len)],
                                               [True] * seq_len)
                },
                # similar to token_classification, it's also a dictionary of task to label
                # SentenceTaskData contains (label, where to extract that label_from)
                # in this case we are going to predict whether a sentence has
                # odd number of tokens or not whenever we see eos token
                sentence_classification={'odd': SentenceTaskData(seq_len % 2, seq_len - 1)}
            )

    # we need eos_id and it's always at this place
    eos_id = vocab_size + TextEncoder.EOS_OFFSET
    # likewise for pad_id
    pad_id = vocab_size + TextEncoder.PAD_OFFSET
    generator = sentence_generator()
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            batch = _create_batch(batch, pad_id, max_len)  # magic to pad and batch sentences
            # at the end it will generate a SentenceBatch which is more than just a list of Sentence
            yield batch
            batch = []

batch_size = 5
generator = tutorial_batch_generator(sentence_piece_encoder.vocab_size, sequence_encoder_config['max_len'],
                                     batch_size, steps=10000)

from transformer.train import train_model

m = train_model(base_model=sequence_encoder, is_causal=False, tasks_meta_data=tasks, pretrain_generator=generator,
                finetune_generator=generator, pretrain_epochs=100, pretrain_steps=number_of_pretrain_steps // 100,
                finetune_epochs=100, finetune_steps=number_of_finetune_steps // 100, verbose=2)
# now m is ready to be used!
print(m.inputs)

import numpy as np

bs = 6
vs = sentence_piece_encoder.vocab_size
sl = sequence_encoder_config['max_len']
# generate random tokens
token = np.random.randint(0, vs, (bs, sl))
# generate random seg_id
segment = np.random.randint(0, 2, (bs, sl))
# generate pos_id
from transformer.train import generate_pos_ids

pos = generate_pos_ids(bs, sl)
# generate attn_mask
from data.dataset import create_attention_mask

# first generate a padding_mask(1 means it is not padded)
pad_mask = np.random.randint(0, 2, (bs, sl)).astype(np.int8)
# create the mask
mask = create_attention_mask(pad_mask=pad_mask, is_causal=False)
# generate target index
target_index = np.random.randint(0, sl, (bs, 1))
res = m.predict([token, segment, pos, mask, target_index], verbose=2)
assert res[0].shape == (bs, sl, vs + TextEncoder.SPECIAL_COUNT)  # lm
assert res[1].shape == (bs, 1, 2)  # odd
assert res[2].shape == (bs, sl, vs + TextEncoder.SPECIAL_COUNT)  # random_lm

# Available pretrained model checkpoints:
#   uncased_L-12_H-768_A-12: uncased BERT base model
#   uncased_L-24_H-1024_A-16: uncased BERT large model
#   cased_L-12_H-768_A-12: cased BERT large model
# BERT_MODEL = 'uncased_L-12_H-768_A-12' #@param {type:"string"}
# download_url = 'https://storage.googleapis.com/bert_models/2018_10_18/{}.zip'.format(BERT_MODEL)
# zip_path = '{}.zip'.format(BERT_MODEL)
# BERT_PRETRAINED_DIR = os.path.realpath(BERT_MODEL)
# print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
BERT_PRETRAINED_DIR ='/Volumes/Work/2019_Work/SQLova/reference/bert/language_model/pre-trained/uncased_L-24_H-1024_A-16'

from transformer.load import load_google_bert
g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR+'/')
g_bert.summary()

print(g_bert.inputs,g_bert.outputs)

# @title Compile keras  model here
from transformer.train import train_model
use_tpu=False
if use_tpu:
  assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime; Maybe you should switch hardware accelerator to TPU for TPU support'
  import tensorflow as tf
  tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  strategy = tf.contrib.tpu.TPUDistributionStrategy(
          tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
  )
  g_bert = tf.contrib.tpu.keras_to_tpu_model(
                      g_bert, strategy=strategy)
g_bert.compile('adam', 'mse')

# serialize model to JSON
model_json = g_bert.to_json()
with open("keras_model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
g_bert.save_weights("keras_model/model.h5")
print("Saved model to disk")

import numpy as np
def random_data_generator():
  while True:
    x1 = np.random.randint(0,30000,(8,512))
    x2 = np.random.randint(0,2,(8,512))
    x3 = np.random.randint(0,512,(8,512))
    x4 = np.random.rand(8,1,512,512)
    y = np.random.rand(8,512,768)
    yield [x1,x2,x3,x4],y

d_g = random_data_generator()

x,y = next(d_g)
print(g_bert.train_on_batch(x,y))
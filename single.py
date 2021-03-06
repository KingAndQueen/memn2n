"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data,character_data
from sklearn import metrics, model_selection
from memn2n import MemN2N

from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import pickle as pkl
import pdb


tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 25, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "my_data_rename", "Directory containing bAbI tasks")
tf.flags.DEFINE_boolean('visual', True, 'whether visualize the embedding')
tf.flags.DEFINE_boolean('joint', False, 'whether to train all tasks')
tf.flags.DEFINE_boolean('trained_emb', False, 'whether use trained embedding, such as Glove')
tf.flags.DEFINE_boolean('introspect', True, 'whether use the introspect unit')

FLAGS = tf.flags.FLAGS

print("Started Task:", FLAGS.task_id)

if FLAGS.joint:
    ids = range(1, 21)
    train, test, train_tags, test_tags = [], [], [], []
    # pdb.set_trace()
    for i in ids:
        tr, te, tr_tag, te_tag = load_task(FLAGS.data_dir, i, joint=True)
        train += tr
        train_tags += tr_tag
        test = te
        test_tags = te_tag
        # pdb.set_trace()

else:
    # task data
    train, test, train_tags, test_tags = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

# pdb.set_trace()
# vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
vocab_my = []
for s, q, a in data:
    sample = list(list(chain.from_iterable(s)) + q + a)
    for word in sample:
        if word not in vocab_my:
            vocab_my.append(word)
vocab = vocab_my
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

# pdb.set_trace()

train_set = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train)))
train_set = [word_idx[id] for id in train_set]

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
memory_size = min(FLAGS.memory_size, max_story_size)

_oov_word = len(train_set)
oov_word_ = len(vocab)
print('oov words', oov_word_ - _oov_word)
# pdb.set_trace()
# Add time words/indexes
# print('vocab word length:',len(word_idx))
for i in range(memory_size):
    word_idx['sents{}'.format(i + 1)] = len(word_idx)+1 #'sents{}'.format(i + 1)
# word_idx['<pad>']=0
print('vocab word+time length:', len(word_idx))


# pdb.set_trace()

vocab_size = len(word_idx) + 1  # +1 for nil word
sentence_size = max(query_size, sentence_size)  # for the position
sentence_size += 1  # +1 for time words

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)

trainS, valS, trainQ, valQ, trainA, valA = model_selection.train_test_split(S, Q, A, test_size=.1,
                                                                            random_state=FLAGS.random_state)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)
# pdb.set_trace()
print(testS[1])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size

batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]

if FLAGS.trained_emb:
    import build_embedding
    word_idx['<pad>'] = 0
    data_path = FLAGS.data_dir + '/vocab.pkl'
    f = open(data_path, 'wb')
    pkl.dump(word_idx, f)
    f.close()
    glove_path = './glove.twitter.27B.25d.txt'
    vocab_g, emb_g = build_embedding.loadGlove(glove_path, emb_size=25)
    print('glove vocab_size', len(vocab_g))
    print('glove embedding_dim', len(emb_g[0]))
    # pdb.set_trace()
    emb, word2idx = build_embedding.idx_to_emb('./my_data_replace/vocab.pkl', emb_size=25)
    emb_new = build_embedding.update_emb(emb, word2idx, vocab_g, emb_g, './my_data_replace/new_embed.pkl')
    my_embedding = pkl.load(open(FLAGS.data_dir + '/new_embed.pkl', 'rb'))
else:
    my_embedding = None

idx_word={value:key for key, value in word_idx.items()}


with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, trained_embedding=FLAGS.trained_emb,
                   _my_embedding=my_embedding)
    for t in range(1, FLAGS.epochs + 1):
        # Stepped learning rate
        if t - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((t - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        np.random.shuffle(batches)
        total_cost = 0.0
        for start, end in batches:
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a, lr)
            total_cost += cost_t

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred, _ = model.predict(s, q)
                train_preds += list(pred)

            val_preds, _ = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            val_acc = metrics.accuracy_score(val_preds, val_labels)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds, word_embedding = model.predict(testS, testQ, type='test')
    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)

    if FLAGS.introspect:
        test_preds, word_embedding_iu = model.predict(testS, testQ, type='introspect', test_tags=test_tags,
                                                  train_data=[S, Q, A, train_tags],
                                                  word_idx=word_idx, train_set=train_set)
        test_acc = metrics.accuracy_score(test_preds, test_labels)
        print("Introspection Testing Accuracy:", test_acc)
    if FLAGS.visual:
        import draw
        draw.drew_embedding(word_embedding,idx_word,name='trained_word_emb')
        draw.drew_embedding(word_embedding_iu, idx_word, name='IU_word_emb')
        # draw.draw_relation(word_embedding[_oov_word:oov_word_], word_embedding_iu[_oov_word:oov_word_])
        # draw.draw_relation(word_embedding, word_embedding_iu)

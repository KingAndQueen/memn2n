"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
import pdb
import copy
import nltk
import re


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        # pdb.set_trace()
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        # z = tf.zeros(shape=[1,s])
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def find_lcseque(s1, s2):
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # pdb.set_trace()
    # print (np.array(d))
    s = []
    while m[p1][p2]:
        c = d[p1][p2]
        if c == 'ok':
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':
            p2 -= 1
        if c == 'up':
            p1 -= 1
    s.reverse()
    return s


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MemN2N(object):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 encoding=position_encoding,
                 session=tf.Session(),
                 name='MemN2N',trained_embedding=False,_my_embedding=None):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name
        self._my_embedding = _my_embedding
        self.trained_embedding =trained_embedding

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._lr)

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._queries)  # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                labels=tf.cast(self._answers, tf.float32),
                                                                name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        # pdb.set_trace()
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            # nil_word_slot = tf.zeros([1, self._embedding_size])
            # A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            # C = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            #

            # pdb.set_trace()
            if self.trained_embedding:
                # self.A_1 = tf.get_variable('embedding_word', shape=[self._vocab_size, self._embedding_size],
                #                                    initializer=tf.constant_initializer(value=self._my_embedding,
                #                                                                        dtype=tf.float32),trainable=True)
                A = tf.random_normal([self._vocab_size, self._embedding_size], stddev=0.1)
                self.A_1 = tf.Variable(A, name="A")
                self.C = []

                for hopn in range(self._hops):
                    with tf.variable_scope('hop_{}'.format(hopn)):
                        self.C.append(tf.get_variable('embedding_word', shape=[self._vocab_size, self._embedding_size],
                                                   initializer=tf.constant_initializer(value=self._my_embedding,
                                                                                       dtype=tf.float32),trainable=True))
            else:
                A = tf.random_normal([self._vocab_size, self._embedding_size], stddev=0.1)
                self.A_1 = tf.Variable(A, name="A")
                C = tf.random_normal([self._vocab_size, self._embedding_size], stddev=0.1)
                self.C = []

                for hopn in range(self._hops):
                    with tf.variable_scope('hop_{}'.format(hopn)):
                        self.C.append(tf.Variable(C, name="C"))

                    # Dont use projection for layerwise weight sharing
                    # self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")

                    # Use final C as replacement for W
                    # self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            # Use A_1 for thee question embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(self.A_1, stories)
                    m_A = tf.reduce_sum(m_emb_A * self._encoding, 2)

                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.C[hopn - 1], stories)
                        m_A = tf.reduce_sum(m_emb_A * self._encoding, 2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                with tf.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn], stories)
                m_C = tf.reduce_sum(m_emb_C * self._encoding, 2)

                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                # Dont use projection layer for adj weight sharing
                # u_k = tf.matmul(u[-1], self.H) + o_k

                u_k = u[-1] + o_k

                # nonlinearity
                # if self._nonlin:
                #     u_k = nonlin(u_k)

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k, tf.transpose(self.C[-1], [1, 0]))

    def batch_fit(self, stories, queries, answers, learning_rate):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers, self._lr: learning_rate}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def simulate_query(self, test_stories, test_queries, test_tags, train_data, word_idx, train_word_set):
        # pdb.set_trace()
        s = train_data[0]
        q = train_data[1]
        a = train_data[2]
        tags_train = train_data[3]
        tags_test = test_tags
        # pdb.set_trace()
        name_map_ = self.entities_map(tags_test, tags_train, s, test_stories, train_word_set)
        # pdb.set_trace()
        name_map = {}
        for test_entity, train_entities in name_map_.items():
            for train_entity in train_entities:
                if train_entity not in name_map.values():
                    name_map[test_entity] = train_entity
                    break
        # pdb.set_trace()
        # if not len(name_map) == len(name_map_): pdb.set_trace()
        name_map = {value: key for key, value in name_map.items()}
        # pdb.set_trace()

        # print('new name_map:', name_map)
        # for query in test_queries:
        #     a_list=test_entities
        #     b_list=query
        #     cross=list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))
        #     if len(cross)>0:
        # pdb.set_trace()
        # print('simulate querying...')

        # losses = 0
        for s_e in range(100):
            losses = self.simulate_train(name_map, s, q, a, 0.01)
            print('The %d th simulation loss:%f' % (s_e, losses))

    def entities_map(self, tags_test, tags_train, train_stories, test_stories, train_set):
        name_map = {}

        # samples=[]
        def similar_sample(tags_test_sent_, tags_train_, position):
            similar_sample_index = []
            longest_len = 0
            for idx_story, tags_story in enumerate(tags_train_):
                for idx_sent, tags_sents in enumerate(tags_story):
                    length = len(find_lcseque(tags_test_sent_, tags_sents))
                    if length>longest_len and len(tags_sents) > position:
                        longest_len = length
                        similar_sample_index=[]
                        similar_sample_index.append([idx_story, idx_sent])
                    if length == longest_len and len(tags_sents) > position:
                        similar_sample_index.append([idx_story, idx_sent])

            return similar_sample_index

        def new_words_position(sent, train_set):
            new_words_p = []
            new_word = []
            # token = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
            for idx, word in enumerate(sent):
                if word not in train_set and not word == 0:
                    new_words_p.append(idx)
                    new_word.append(word)
            return new_words_p, new_word

        for idx_story, story in enumerate(test_stories):
            # print('test number:', idx_story)
            for idx_sents, sents in enumerate(story):
                # pdb.set_trace()
                recognise = False
                position_list, new_words = new_words_position(sents[:-1], train_set)
                for words in new_words:
                    if words not in name_map.keys():
                        recognise = True
                # print (recognise,new_words,name_map)
                if len(position_list) > 0 and recognise:
                    for position in position_list:
                        similar_smaple_in_train_positions = similar_sample(tags_test[idx_story][idx_sents], tags_train, position)
                        # pdb.set_trace()
                        for train_position in similar_smaple_in_train_positions:
                          try:
                            if tags_train[train_position[0]][train_position[1]][position] == \
                                    tags_test[idx_story][idx_sents][position]:
                                # pdb.set_trace()
                                value = train_stories[train_position[0]][train_position[1]][position]
                                if sents[position] not in name_map.keys():
                                    name_map[sents[position]] = [value]
                                elif value not in name_map[sents[position]]:
                                    name_map[sents[position]].append(value)
                          except:
                                pdb.set_trace()

        return name_map

    def simulate_train(self, name_map, story, query, answer, lr):
        stories, queries, answers = [], [], []
        # for key,value in name_map.items():
        #     name_map_temp={value:key}
        flag = False
        for i in range(len(query)):
            s = copy.copy(story[i])
            q = copy.copy(query[i])
            a = copy.copy(answer[i])
            for no, id in enumerate(q):
                if id in name_map.keys():
                    q[no] = name_map[id]
                    flag = True
            for _no, sent in enumerate(s):
                for no_, id in enumerate(sent):
                    if id in name_map.keys():
                        sent[no_] = name_map[id]
                        flag = True
            # pdb.set_trace()

            ans_id = np.argmax(a, 0)
            if ans_id in name_map.keys():
                a = np.zeros(len(a))
                a[name_map[ans_id]] = 1
                flag = True
            # pdb.set_trace()
            if flag:
                stories.append(s)
                queries.append(q)
                answers.append(a)
                flag = False

        if len(queries) <= 0: pdb.set_trace()
        total_cost = 0.0
        if len(queries) > 32:
            batches = zip(range(0, len(queries) - 32, 32), range(32, len(queries), 32))
            batches = [(start, end) for start, end in batches]
            np.random.shuffle(batches)
            # pdb.set_trace()
            for start, end in batches:
                s = stories[start:end]
                q = queries[start:end]
                a = answers[start:end]
                cost_t = self.batch_fit(s, q, a, lr)
                total_cost += cost_t
        else:
            total_cost = self.batch_fit(stories, queries, answers, lr)
        return total_cost

    def predict(self, stories, queries, type=None, test_tags=None, train_data=None, word_idx=None, train_set=None):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        if type == 'introspect':
            self.simulate_query(stories, queries, test_tags, train_data, word_idx, train_set)
            feed_dict = {self._stories: stories, self._queries: queries}
            return self._sess.run([self.predict_op, self.A_1], feed_dict=feed_dict)
        else:
            feed_dict = {self._stories: stories, self._queries: queries}
            return self._sess.run([self.predict_op, self.A_1], feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)

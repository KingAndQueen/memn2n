from __future__ import absolute_import

import os
import re
import numpy as np
import nltk,pdb
import copy
def load_task(data_dir, task_id, joint=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    if not joint:
        test_file = [f for f in files if s in f and 'test' in f][0]
    else:
        test_file = [f for f in files if 'joint_test' in f][0]
    train_data,tags_train = get_stories(train_file,)
    test_data,tags_test = get_stories(test_file,)
    return train_data, test_data, tags_train,tags_test

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    tags_data=[]
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            # pdb.set_trace()
            tags = []
            for sent in substory:
                tag_list=[]
                tag_sents=nltk.pos_tag(sent)
                for tag in tag_sents:
                    tag_list.append(tag[1])
                # if tag_list not in tags:
                tags.append(tag_list)
            tags_data.append(tags[-50:])
            # pdb.set_trace()
            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    # pdb.set_trace()
    return data, tags_data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        data,tags=parse_stories(f.readlines(), only_supporting=only_supporting)
        return data,tags

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which 
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


def character_data(data, word_idx,idx_word, sentence_size, memory_size,character_len=10):
    S = []
    Q = []
    A = []

    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            sents_char=[]
            for w in sentence:
                word_char = []
                for ch in w:
                    word_char.append(ord(ch))
                ls_c=max(0,character_len-len(word_char))
                for _ in range(ls_c):
                    word_char.append(0)
                sents_char.append(copy.copy(word_char))

            for _ in range(ls):
                sents_char.append([0]*character_len)
            ss.append(copy.copy(sents_char))

        # pdb.set_trace()
        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            sents_seq = len(word_idx) - memory_size - i + len(ss)
            sents_seq_char=idx_word[sents_seq]
            sents_seq_chars=[]
            for c in sents_seq_char:
                sents_seq_chars.append(ord(c))
            ls_c = max(0, character_len - len(sents_seq_chars))
            for _ in range(ls_c):
                sents_seq_chars.append(0)
            ss[i][-1]=copy.copy(sents_seq_chars)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            word_pad=[0]*character_len
            sent_pad=[]
            for _sp in range(sentence_size):
                sent_pad.append(word_pad)
            ss.append(sent_pad)

        lq = max(0, sentence_size - len(query))
        # q = [word_idx[w] for w in query] + [0] * lq
        q=[]
        for w in query:
            word_char = []
            for ch in w:
                word_char.append(ord(ch))
            ls_c = max(0, character_len - len(word_char))
            for _ in range(ls_c):
                word_char.append(0)
            q.append(copy.copy(word_char))
        for _ in range(lq):
            q.append([0] * character_len)

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(np.array(ss))
        Q.append(q)
        A.append(y)
    # pdb.set_trace()
    return np.array(S), np.array(Q), np.array(A)
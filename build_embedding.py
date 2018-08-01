import numpy as np
import pdb
import os
#import Multi_Roles_Data
import pickle as pkl

def idx_to_emb(pickle_path,emb_size=100):
# bulid 'number to embedding' vocabulary
    if not os.path.exists(pickle_path):
        print('vocab path not exist!')
    else:
        word2idx = pkl.load(open(pickle_path, 'rb'))
        # word2idx=vocab_class.word2idx
        # idx2word=vocab_class.idx2word
        vocab_size=len(word2idx)
        print('old vocab size:',vocab_size)
        emb=np.random.normal(size=(vocab_size,emb_size))
        return emb,word2idx


def update_emb(emb,word2idx,vocab_glove,emb_glove,pickle_path='my_embedding.pkl'):
        count=0
        count_unk=0
        for word, idx in word2idx.items():
            # print('%d / %d' % (count,len(word2idx)) )
            word=word.lower()
            count+=1
       #     pdb.set_trace()
            if word in vocab_glove:
                idx_g = vocab_glove.index(word)
                emb_g=emb_glove[idx_g]
                # pdb.set_trace()
                emb[idx]=emb_g
            else:
                count_unk+=1
                # print('word not in glove',word)
        print('unfind word: ',count_unk)
        pkl.dump(emb, open(pickle_path, 'wb'))

        return emb

def loadGlove(filename,emb_size=100):
    vocab = []
    embd = []
    vocab.append('unk')
    embd.append([0]*emb_size)
    file = open(filename,'rb')
    print('Loaded GloVe!')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0].decode('utf-8'))
        vec=row[1:]
        vec=np.array(vec)
        vec=vec.astype('float32')
        embd.append(vec)
    file.close()
  #  embd=np.array(embd)
   # embd=embd.astype('float32')
    return vocab,embd

if __name__=='__main__':
    glove_path='./glove.twitter.27B.25d.txt'
    vocab_g,emb_g=loadGlove(glove_path,emb_size=25)
    print('glove vocab_size' , len(vocab_g))
    print('glove embedding_dim', len(emb_g[0]))
    # pdb.set_trace()
    emb,word2idx=idx_to_emb('./my_data_replace/vocab.pkl',emb_size=25)
    emb_new=update_emb(emb,word2idx,vocab_g,emb_g,'./my_data_replace/new_embed.pkl')
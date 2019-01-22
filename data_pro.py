import numpy as np
import logging
from collections import Counter


def load_data(file):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = line.strip().lower().split()
            relations.append(int(line[0]))
            e1_pos.append((int(line[1]), int(line[2])))  # (start_pos, end_pos)
            e2_pos.append((int(line[3]), int(line[4])))  # (start_pos, end_pos)
            sentences.append(line[5:])

    return sentences, relations, e1_pos, e2_pos


def build_dict(sentences):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1  # count all words and its frequencies

    ls = word_count.most_common()
    word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}  # w[0]: use the head word?
    # leave 0 to PAD
    return word_dict        # a dict of words and its frequencies


def load_embedding(emb_file, emb_vocab, word_dict):
    vocab = {}
    with open(emb_vocab, 'r') as f:
        for id, w in enumerate(f.readlines()):
            w = w.strip().lower()
            vocab[w] = id   # vocab of the pre-trained embedding

    f = open(emb_file, 'r')
    embed = f.readlines()

    dim = len(embed[0].split())
    num_words = len(word_dict) + 1  # num_words of the corpus
    embeddings = np.random.uniform(-0.01, 0.01, size=(num_words, dim))

    n_pre_trained = 0
    for w in vocab.keys():
        if w in word_dict:
            embeddings[word_dict[w]] = [float(x) for x in embed[vocab[w]].split()]
            n_pre_trained += 1
    embeddings[0] = np.zeros(dim)

    logging.info(
        'embeddings: %.2f%%(pre_trained) unknown: %d' % (n_pre_trained / num_words * 100, num_words - n_pre_trained))

    f.close()
    return embeddings.astype(np.float32)


# Re-implementation of load_embedding()
def load_glove_embeddings(emb_file, word_dict):
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    tmp_file = emb_file + '.vec'
    glove2word2vec(emb_file, tmp_file)
    word2vec_model = KeyedVectors.load_word2vec_format(tmp_file, binary=False, encoding='utf-8', unicode_errors='ignore')

    dim = 300
    num_words = len(word_dict) + 1  # num_words of the corpus
    embedding_matrix = np.random.uniform(-0.01, 0.01, size=(num_words, dim))

    n_pre_trained = 0
    for w in word_dict:
        if w in word2vec_model:
            embedding_matrix[word_dict[w]:] = word2vec_model[w]
            n_pre_trained += 1
    logging.info(
        'embeddings: %.2f%%(pre_trained) unknown: %d' % (n_pre_trained / num_words * 100, num_words - n_pre_trained))
    return embedding_matrix.astype(np.float32)


def pos(x):
    '''
    map the relative distance between [0, 123)
    '''
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122


def vectorize(data, word_dict, max_len):
    sentences, relations, e1_pos, e2_pos = data
    n_instance = len(sentences)
    # print('Max seq len:', max([len(sent) for sent in sentences]))     # 97

    # replace word with word-id
    e1_vec = []
    e2_vec = []
    # sents_vec = []
    sents_vec = np.zeros((n_instance, max_len), dtype=int)
    logging.debug('data shape: (%d, %d)' % (n_instance, max_len))

    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]  # word to word-id
        sents_vec[idx, :len(vec)] = vec

        # # log e1 and e2 if e1 or e2 is a phrase
        # if pos1[0]!=pos1[1] or pos2[0]!=pos2[1]:
        #   s_e1 = ''
        #   for w in sent[pos1[0] : pos1[1]+1]:
        #     s_e1 += w + ' '
        #   s_e2 = ''
        #   for w in sent[pos2[0] : pos2[1]+1]:
        #     s_e2 += w + ' '
        #   logging.debug("%s - %s" % (s_e1, s_e2))

        # # the entire e1 and e2 phrase
        # e1_vec.append(vec[pos1[0] : pos1[1]+1])
        # e2_vec.append(vec[pos2[0] : pos2[1]+1])

        # last word of e1 and e2
        e1_vec.append(vec[pos1[1]])     # pos1[1]: end position of e1
        e2_vec.append(vec[pos2[1]])

    # compute relative distance
    dist1 = []
    dist2 = []
    for sent, e1_p, e2_p in zip(sents_vec, e1_pos, e2_pos):
        # current word position - last word position of e1 or e2
        dist1.append([pos(current_p - e1_p[1]) for current_p, _ in enumerate(sent)])
        dist2.append([pos(current_p - e2_p[1]) for current_p, _ in enumerate(sent)])

        # dist1.append([(current_p - e1_p[1]) for current_p, _ in enumerate(sent)])
        # dist2.append([(current_p - e2_p[1]) for current_p, _ in enumerate(sent)])
    # print('dist1: len={}, max={}, min={}'.format(len(dist1), max([max(d) for d in dist1]), min([min(d) for d in dist1])))
    # print('dist2: len={}, max={}, min={}'.format(len(dist2), max([max(d) for d in dist2]), min([min(d) for d in dist2])))

    # return sents_vec, relations, e1_vec, e2_vec, dist1, dist2
    return sents_vec, relations, dist1, dist2


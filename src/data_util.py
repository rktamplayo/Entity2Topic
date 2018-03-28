import logging

import numpy as np

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


def load_dict(dict_path, max_vocab=None):
    logging.info("Try load dict from {}.".format(dict_path))
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        logging.info(
            "Load dict {dict} failed, create later.".format(dict=dict_path))
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    logging.info(
        "Load dict {} with {} words.".format(dict_path, len(tok2id)))
    return (tok2id, id2tok)


def create_dict(dict_path, corpus, max_vocab=None):
    logging.info("Create dict {}.".format(dict_path))
    counter = {}
    for line in corpus:
        for word in line:
            if word == MARK_PAD:
                continue
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    logging.info(
        "Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)


def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk += 1
        ret.append(tmp)
    return ret, (tot - unk) / tot


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


def load_data(doc_filename,
              sum_filename,
              doc_dict_path,
              sum_dict_path,
              en_filename,
              en_dict_path,
              word_vec_path,
              en_vec_path,
              max_doc_vocab=None,
              max_sum_vocab=None,
              max_en_vocab=None):
    logging.info(
        "Load document from {}; summary from {}.".format(
            doc_filename, sum_filename))

    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    with open(sum_filename) as sumfile:
        sums = sumfile.readlines()
    #entity
    with open(en_filename) as enfile:
        entitys = enfile.readlines()
    assert len(docs) == len(sums)
    logging.info("Load {num} pairs of data.".format(num=len(docs)))

    docs = list(map(lambda x: x.split(), docs))
    sums = list(map(lambda x: x.split(), sums))
    #entity
    entitys = list(map(lambda x: x.split(), entitys))
    entitys = [x for x in entitys if x != '']
    
    doc_dict = load_dict(doc_dict_path, max_doc_vocab)
    if doc_dict is None:
        doc_dict = create_dict(doc_dict_path, docs, max_doc_vocab)

    sum_dict = load_dict(sum_dict_path, max_sum_vocab)
    if sum_dict is None:
        sum_dict = create_dict(sum_dict_path, sums, max_sum_vocab)
    #entity
    en_dict = load_dict(en_dict_path, max_en_vocab)
    if en_dict is None:
        en_dict = create_dict(en_dict_path, entitys, max_en_vocab)

    word_vectors = {}
    f = open(word_vec_path, 'r')
    for line in f:
        line = line.split()
        key = line[0]
        vec = [float(x) for x in line[-300:]]
        word_vectors[key] = vec
    f.close()
    doc_vec = []
    doc_id2tok = doc_dict[1]
    doc_cnt = 0
    for i in range(len(doc_id2tok)):
        key = doc_id2tok[i]
        if key in word_vectors:
            doc_cnt += 1
            doc_vec.append(np.array(word_vectors[key]))
        else:
            doc_vec.append(np.random.uniform(-0.25, 0.25, 300))
    doc_vec = np.array(doc_vec)
    print('Available document vectors:', doc_cnt)
    sum_vec = []
    sum_id2tok = sum_dict[1]
    sum_cnt = 0
    for i in range(len(sum_id2tok)):
        key = sum_id2tok[i]
        if key in word_vectors:
            sum_cnt += 1
            sum_vec.append(np.array(word_vectors[key]))
        else:
            sum_vec.append(np.random.uniform(-0.25, 0.25, 300))
    sum_vec = np.array(sum_vec)
    print('Available summary vectors:', sum_cnt)
    
    vectors = {}
    f = open(en_vec_path, 'r')
    for line in f:
        line = line.split()
        key = line[0]
        vec = [float(x) for x in line[1:]]
        vectors[key] = vec
    f.close()
    en_vec = []
    en_id2tok = en_dict[1]
    en_cnt = 0
    for i in range(len(en_id2tok)):
        key = en_id2tok[i]
        if key in vectors:
            en_cnt += 1
            en_vec.append(np.array(vectors[key]))
        else:
            en_vec.append(np.random.uniform(-0.25, 0.25, 1000))
    en_vec = np.array(en_vec)
    print('Available entity vectors:', en_cnt)
    
    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words.".format(cover * 100))
    sumid, cover = corpus_map2id(sums, sum_dict[0])
    logging.info(
        "Sum dict covers {:.2f}% words.".format(cover * 100))
    #entity
    enid, cover = corpus_map2id(entitys, en_dict[0])

    return docid, sumid, doc_dict, sum_dict, enid, en_dict, doc_vec, sum_vec, en_vec


def load_valid_data(doc_filename,
                    sum_filename,
                    doc_dict,
                    sum_dict,
                    en_filename,
                    en_dict):
    logging.info(
        "Load validation document from {}; summary from {}.".format(
            doc_filename, sum_filename))
    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    with open(sum_filename) as sumfile:
        sums = sumfile.readlines()
    #entity
    with open(en_filename) as enfile:
        entitys = enfile.readlines()
    assert len(sums) == len(docs)

    logging.info("Load {} validation documents.".format(len(docs)))

    docs = list(map(lambda x: x.split(), docs))
    sums = list(map(lambda x: x.split(), sums))
    #entity
    entitys = list(map(lambda x: x.split(), entitys))

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words on validation set.".format(cover * 100))
    sumid, cover = corpus_map2id(sums, sum_dict[0])
    logging.info(
        "Sum dict covers {:.2f}% words on validation set.".format(cover * 100))
    #entity
    enid, cover = corpus_map2id(entitys, en_dict[0])
    return docid, sumid, enid


def corpus_preprocess(corpus):
    import re
    ret = []
    for line in corpus:
        x = re.sub('\\d', '#', line)
        ret.append(x)
    return ret


def sen_postprocess(sen):
    return sen


def load_test_data(doc_filename, doc_dict, en_filename, en_dict):
    logging.info("Load test document from {doc}.".format(doc=doc_filename))

    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    docs = corpus_preprocess(docs)
    with open(en_filename) as enfile:
        entitys = enfile.readlines()
    
    logging.info("Load {num} testing documents.".format(num=len(docs)))
    docs = list(map(lambda x: x.split(), docs))
    
    entitys = list(map(lambda x: x.split(), entitys))

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info(
        "Doc dict covers {:.2f}% words.".format(cover * 100))
        
    #entity
    enid, cover = corpus_map2id(entitys, en_dict[0])

    return docid, enid


if __name__ == "__main__":
    logging.basicConfig(
                        level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    docid, sumid, doc_dict, sum_dict = load_data(
        "data/train.article.txt", "data/train.title.txt",
        "data/doc_dict.txt", "data/sum_dict.txt",
        "data/train.entity.txt", "data/en_dict.txt",
        25000, 25000)

    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], sen_map2tok(sumid[checkid], sum_dict[1]))

    docid, sumid, enid = load_valid_data(
        "data/valid.article.filter.txt", "data/valid.title.filter.txt",
        doc_dict, sum_dict, "data/valid.entity.txt")

    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], sen_map2tok(sumid[checkid], sum_dict[1]))

    docid = load_test_data("data/test.giga.txt", doc_dict)
    checkid = np.random.randint(len(docid))
    print(checkid)
    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))

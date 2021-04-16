import os
import pickle
import h5py
import numpy as np
from numpy import linalg as LNG
from string import punctuation
from collections import Counter

from nltk import word_tokenize, PorterStemmer

from LinkedList import LinkedList

PUNCTUATION = punctuation + '-—'
PORTER = PorterStemmer()
INPUT_DIR = 'HillaryEmails'
FILE_TERM_DOC_PAIRS = 'output.txt'
SAVE = False  # Save inverted_index else load


def save_w():
    N = len(docId_to_doc)
    all_terms = [x[2] for x in inverted_index]
    tfs = np.zeros(shape=(len(all_terms), len(docId_to_doc)), dtype=np.int)
    dfs = np.zeros(shape=(len(all_terms)), dtype=np.int)
    w = np.zeros(shape=(len(all_terms), len(docId_to_doc)), dtype=np.float)
    for docId, doc in docId_to_doc.items():
        doc = os.path.join(INPUT_DIR, doc)
        with open(doc, 'rt', encoding='utf-8') as f:
            texts = f.read()
            terms = preprocess(texts)
            terms = Counter(terms)
            for term in terms:
                index = search(term)
                if index is None:
                    print('Error! {} not found!'.format(term))
                    # exit(-13)
                    continue
                tfs[index, docId] = terms[term]
                dfs[index] += 1

    vfunc = np.vectorize(lambda x: 1 if x <= 0 else x)
    dfs = vfunc(dfs)

    tfs = np.log10(tfs + 1)
    dfs = np.log10(N / dfs)

    rows = w.shape[0]
    cols = w.shape[1]
    for x in range(rows):
        for y in range(cols):
            w[x, y] = tfs[x, y] * dfs[x]

    lengths = LNG.norm(w, axis=0)

    h5f = h5py.File('w2.h5', 'w')
    h5f.create_dataset('w', data=w)
    h5f.create_dataset("lengths", data=lengths)
    h5f.close()


def load_w():
    h5f = h5py.File('w.h5', 'r')
    w = h5f.get('w')[()]
    lengths = h5f.get('lengths')[()]
    return w, lengths


def get_tfidf(t):
    pass


def save_data():
    with open('saved_data.pkl', 'wb') as f:
        dict_cp = []
        for i in range(len(inverted_index)):
            dict_cp.append([inverted_index[i][0], inverted_index[i][1].to_list(), inverted_index[i][2]])
        pickle.dump((dict_cp, docId_to_doc, doc_to_docId), f)


def load_data():
    with open('saved_data.pkl', 'rb') as f:
        inverted_index, docId_to_doc, doc_to_docId = pickle.load(f)
        for i in range(len(inverted_index)):
            inverted_index[i][1] = LinkedList.from_list(inverted_index[i][1])
    return inverted_index, docId_to_doc, doc_to_docId


def search(term):
    a = 0
    b = len(inverted_index)
    i = int((a + b) / 2)
    while True:
        ptr = inverted_index[i][2]
        if ptr == term:
            return i
        elif ptr < term:
            a = i + 1
        else:
            b = i - 1
        if a == b:
            i = a
            continue
        elif a > b:
            return None
        i = int((a + b) / 2)


def preprocess(terms):
    terms = word_tokenize(terms)
    sz = len(terms)
    i = 0
    while i < sz:
        j = 0
        # Remove the terms that consist only punctuations
        for x in terms[i]:
            if x not in PUNCTUATION:
                break
            else:
                j += 1
        else:
            # All characters are punctuations, remove the term
            del terms[i]
            sz -= 1
            continue

        if j != 0:
            # Remove prefix punctuations
            terms[i] = terms[i][j:]

        terms[i] = terms[i].lower()
        terms[i] = PORTER.stem(terms[i])

        i += 1
    return terms


def search(term):
    a = 0
    b = len(inverted_index)
    i = int((a + b) / 2)
    while True:
        ptr = inverted_index[i][2]
        if ptr == term:
            return i
        elif ptr < term:
            a = i + 1
        else:
            b = i - 1
        if a == b:
            i = a
            continue
        elif a > b:
            return None
        i = int((a + b) / 2)


def query(terms, k):
    terms = set(preprocess(terms))
    N = len(docId_to_doc)
    scores = [[i, 0] for i in range(N)]
    for term in terms:
        termId = search(term)
        if termId is None:
            print('{} not found'.format(term))
            continue
        ll = inverted_index[termId][1]
        if ll is None or len(ll) == 0:
            print('No matched document was found for term {}'.format(term))
        p = ll.head
        while p:
            docId = p.data
            scores[docId][1] += w[termId, docId]
            p = p.next
    for i in range(N):
        length = lengths[i]
        if length != 0:
            scores[i][1] /= length

    scores.sort(key=lambda x: -x[1])
    scores = scores[:k]
    print("Top {} documents:".format(k))
    print('+************+************+')
    print('*    Doc    *     Score   *')
    print('+************+************+')
    for x in scores:
        print(f"*{'':2}{docId_to_doc[x[0]]:8}{'':2}*{'':3}{x[1]:6.4f}{'':3}*")


if SAVE:
    inverted_index = []
    docId_to_doc = {}
    doc_to_docId = {}
    doc_index = 0
    cur_term = ''
    cur_termId = -1
    prev_doc = ''

    for dirpath, dirnames, filenames in os.walk(INPUT_DIR):
        filenames.sort()
        for f in filenames:
            docId_to_doc[doc_index] = f
            doc_to_docId[f] = doc_index
            doc_index += 1

    with open(FILE_TERM_DOC_PAIRS, 'rt', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line == '':
                # EOF
                break

            pair = line.split()
            term = pair[0]
            doc = pair[1]

            if term != cur_term:
                # Frequency & Postings & Term
                inverted_index.append([0, LinkedList(), term])
                cur_term = term
                cur_termId += 1
                prev_doc = ''

            if doc != prev_doc:
                # Update inverted index, make sure no repeated doc
                inverted_index[cur_termId][0] += 1
                inverted_index[cur_termId][1].append(doc_to_docId[doc])
            prev_doc = doc
    save_data()
    save_w()
else:
    inverted_index, docId_to_doc, doc_to_docId = load_data()
    w, lengths = load_w()

### Sample queries ###
# query('libya missile Gaddafi', 10)

cmd = input('Enter your query: ')
while cmd != '\\q':
    query(cmd, 10)
    print()
    cmd = input('Enter your query: ')

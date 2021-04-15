import os
import pickle
from string import punctuation

from nltk import word_tokenize, PorterStemmer

from LinkedList import LinkedList

PUNCTUATION = punctuation + '-—'
PORTER = PorterStemmer()
INPUT_DIR = 'HillaryEmails'
FILE_TERM_DOC_PAIRS = 'output.txt'
SAVE = False  # Save inverted_index else load


def output(ll):
    if ll is None or len(ll) == 0:
        print('No matched document was found')
        return

    p = ll.head
    print('Found {} documents:'.format(len(ll)))
    while p:
        print(docId_to_doc[p.data])
        p = p.next


def calc_size():
    from sys import getsizeof
    sz = 0
    for x in inverted_index:
        sz += 2  # Frequency in 2 bytes
        sz += (6 + len(x[1]) * (4 + 3))  # 2 pointers (head & tail), data & next pointer for each element
        sz += getsizeof(x[2])
    return sz


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


def intersect(l1, l2):
    p1 = l1.head
    p2 = l2.head
    ret = LinkedList()
    while p1 and p2:
        if p1.data == p2.data:
            ret.append(p1.data)
            p1 = p1.next
            p2 = p2.next
        elif p1.data < p2.data:
            p1 = p1.next
        else:
            p2 = p2.next
    return ret


def merge(l1, l2):
    p1 = l1.head
    p2 = l2.head
    ret = LinkedList()
    while p1 and p2:
        if p1.data == p2.data:
            ret.append(p1.data)
            p1 = p1.next
            p2 = p2.next
        elif p1.data < p2.data:
            ret.append(p1.data)
            p1 = p1.next
        else:
            ret.append(p2.data)
            p2 = p2.next
    while p1:
        ret.append(p1.data)
        p1 = p1.next
    while p2:
        ret.append(p2.data)
        p2 = p2.next
    return ret


def inverse(l):
    ret = LinkedList()
    n = len(docId_to_doc) - 1
    p = l.head
    i = 0
    while p and i <= n:
        if i < p.data:
            ret.append(i)
        else:
            p = p.next
        i += 1
    while i <= n:
        ret.append(i)
        i += 1

    return ret


def query(option, terms):
    answer = None
    terms = set(preprocess(terms))
    indexes = []
    for term in terms:
        result = search(term)
        if result is not None:
            indexes.append(result)
    indexes.sort(key=lambda i: inverted_index[i][0])
    if option == 'and':
        if len(indexes) < 2 or len(indexes) < len(terms):
            # Only one matched or some terms not found
            return None
        l1 = inverted_index[indexes.pop(0)][1]
        l2 = inverted_index[indexes.pop(0)][1]
        answer = intersect(l1, l2)
        while indexes:
            answer = intersect(answer, inverted_index[indexes.pop(0)][1])
    elif option == 'or':
        if len(indexes) < 1:
            # No matched
            return None
        elif len(indexes) == 1:
            # Skip merging
            return inverted_index[indexes.pop(0)][1]
        l1 = inverted_index[indexes.pop(0)][1]
        l2 = inverted_index[indexes.pop(0)][1]
        answer = merge(l1, l2)
        while indexes:
            answer = merge(answer, inverted_index[indexes.pop(0)][1])
    elif option == 'not':
        if len(terms) != 1:
            return None
        answer = inverse(inverted_index[indexes.pop(0)][1])

    return answer


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
else:
    inverted_index, docId_to_doc, doc_to_docId = load_data()

# calc_size()

### Sample queries ###
# answer = query('and', 'cat dog')
answer = query('and', 'libya missile Gaddafi')
# answer = query('or', 'cat dog')
# answer = query('not', 'the')

output(answer)

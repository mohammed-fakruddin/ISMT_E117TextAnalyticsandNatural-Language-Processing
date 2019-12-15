import numpy as np
from bert_serving.client import BertClient


def get_bert_scores(query_string, bertClient, bert_vec):
    topk = 5
    query_vec = bertClient.encode([query_string])[0]
    score = np.sum(query_vec * bert_vec, axis=1) / np.linalg.norm(bert_vec, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    scores_with_idx=[]
    for idx in topk_idx:
        #print('> %s\t%s' % (score[idx], fund_objectives[idx]))
        print('BERT:', score[idx])
        scores_with_idx.append([idx,score[idx]])
    return scores_with_idx

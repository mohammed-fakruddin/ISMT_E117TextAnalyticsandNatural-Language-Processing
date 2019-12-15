import numpy as np
from pythonModules import preProcess

def get_fastText_scores(query_string, ft_model, data_cleaned):
    topk = 5
    query_string_clean_tokens = preProcess.pre_process_sent(query_string)[0]
    query_string_clean_tokens_sent = " ".join(query_string_clean_tokens)
    scores=[]
    for i, d in enumerate(data_cleaned):
        fund_obj_cleaned=" ".join(d) #fast text expects cleaned sentence
        #print(fund_obj_cleaned)
        score = ft_model.similarity(query_string_clean_tokens_sent, fund_obj_cleaned)
        scores.append(score)

    topk_idx = np.argsort(scores)[::-1][:topk]
    scores_with_idx=[]

    for idx in topk_idx:
        #print('> %s\t%s' % (score[idx], fund_objectives[idx]))
        print('FastText scores are:', scores[idx])
        scores_with_idx.append([idx,scores[idx]])
    return scores_with_idx

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pythonModules import preProcess

def trainWord2VecModel(data_cleaned):

    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(data_cleaned)]
    max_epochs = 10
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

    model.build_vocab(tagged_data)


    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    return model

def get_word2Vec_scores(search_query, model):
  #model = word2VecModel
  #test_data = search_query.lower()
  test_data = preProcess.pre_process_sent(search_query)
  test_data_str = ' '.join(w for w in test_data[0])
  v1 = model.infer_vector([test_data_str])
  model.docvecs.most_similar([v1], topn = 1)
  #idx = model.docvecs.most_similar([v1], topn = 1)[0][0]
  #score = model.docvecs.most_similar([v1], topn = 1)[0][1]
  #print(idx,score)
  return model.docvecs.most_similar([v1], topn = 4)

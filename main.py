from bert_model.load_model import load_bert_model

# load model
bert_preprocess_model, bert_model = load_bert_model()
# trials
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

bert_results = bert_model(text_preprocessed)


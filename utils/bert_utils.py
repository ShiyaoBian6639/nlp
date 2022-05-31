from bert_model.load_model import load_bert_model

# load model
bert_preprocess_model, bert_model = load_bert_model()


def get_sentence_embedding(sentence: str):
    text_preprocessed = bert_preprocess_model(sentence)
    embedding = bert_model(text_preprocessed)["pooled_output"].numpy()
    return embedding

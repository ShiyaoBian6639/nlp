from bert_model.bert_config import tfhub_handle_preprocess, tfhub_handle_encoder
import tensorflow_hub as hub
import tensorflow_text


def load_bert_model():
    # bert preprocess
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    # bert model
    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    return bert_preprocess_model, bert_model

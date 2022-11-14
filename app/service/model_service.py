import json

import requests
from nltk import tokenize
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow import convert_to_tensor
from flask import current_app as app

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

serving_base_url = 'http://sdg-serving-{}:8501/v1'

goal_names = {
    "Goal 1": "No poverty",
    "Goal 2": "Zero hunger",
    "Goal 3": "Good health and well-being",
    "Goal 4": "Quality Education",
    "Goal 5": "Gender equality",
    "Goal 6": "Clean water and sanitation",
    "Goal 7": "Affordable and clean energy",
    "Goal 8": "Decent work and economic growth",
    "Goal 9": "Industry, innovation and infrastructure",
    "Goal 10": "Reduced inequalities",
    "Goal 11": "Sustainable cities and communities",
    "Goal 12": "Responsible consumption and production",
    "Goal 13": "Climate action",
    "Goal 14": "Life below water",
    "Goal 15": "Life in Land",
    "Goal 16": "Peace, Justice and strong institutions",
    "Goal 17": "Partnerships for the goals"
}

model_names = {
    "aurora-sdg": "Aurora SDG classifier - single model",
    "aurora-sdg-multi": "Aurora SDG classifier - multi model",
    "osdg": "OSDG classifier",
    "elsevier-sdg-multi": "Elsevier SDG classifier"
}

classifier_host = 'aurora-sdg-classifier.uni-due.de'


def tokenize_abstracts(abstracts):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts = []
    for abstract in abstracts:
        t_abstract = "[CLS] "
        for sentence in tokenize.sent_tokenize(abstract):
            t_abstract = t_abstract + sentence + " [SEP] "
        t_abstracts.append(t_abstract)
    return t_abstracts


def b_tokenize_abstracts(t_abstracts, max_len=512):
    """Tokenizes sentences with the help
    of a 'bert-base-multilingual-uncased' tokenizer.
    """
    b_t_abstracts = [tokenizer.tokenize(_)[:max_len] for _ in t_abstracts]
    return b_t_abstracts


def convert_to_ids(b_t_abstracts):
    """Converts tokens to its specific
    IDs in a bert vocabulary.
    """
    input_ids = [tokenizer.convert_tokens_to_ids(_) for _ in b_t_abstracts]
    return input_ids


def tokenize_abstract(abstract):
    """For a given texts, adds '[CLS]' and '[SEP]' tokens
    at the beginning and the end of each sentence, respectively.
    """
    t_abstracts = []
    t_abstract = "[CLS] "
    for sentence in tokenize.sent_tokenize(abstract):
        t_abstract = t_abstract + sentence + " [SEP] "
    t_abstracts.append(t_abstract)
    return t_abstract


def abstracts_to_ids(abstracts):
    """Tokenizes abstracts and converts
    tokens to their specific IDs
    in a bert vocabulary.
    """
    tokenized_abstracts = tokenize_abstracts(abstracts)
    b_tokenized_abstracts = b_tokenize_abstracts(tokenized_abstracts)
    ids = convert_to_ids(b_tokenized_abstracts)
    return ids


def pad_ids(input_ids, max_len=512):
    """Padds sequences of a given IDs.
    """
    p_input_ids = pad_sequences(input_ids,
                                maxlen=max_len,
                                dtype="long",
                                truncating="post",
                                padding="post")
    return p_input_ids


def create_attention_masks(inputs):
    """Creates attention masks
    for a given seuquences.
    """
    masks = []
    for seq in inputs:
        seq_mask = [i > 0 for i in seq]
        masks.append(seq_mask)
    return tf.cast(masks, tf.int32)


def prepare_input(abstracts):
    ids = abstracts_to_ids(abstracts)
    padded_ids = pad_ids(ids)
    masks = create_attention_masks(padded_ids)
    return convert_to_tensor(padded_ids), convert_to_tensor(masks)


def run_all_models(abstracts):
    input_ids, masks = prepare_input(abstracts=abstracts)
    predictions = []
    for index in range(len(abstracts)):
        individual_predictions = []
        input = {
            'input_ids': input_ids.numpy().tolist()[index],
            'attention_masks': masks.numpy().tolist()[index]
        }
        data = json.dumps({'instances': [input]})
        headers = {"content-type": "application/json"}
        for index in range(17):

            sdg_number = index + 1
            serving_url = serving_base_url.format(sdg_number)
            url = f'{serving_url}/models/sdg{sdg_number}:predict'
            test = requests.get(f'{serving_url}/models/sdg{sdg_number}')
            app.logger.info(f'url_test for {url}: response = {test.text}')
            response = requests.post(url, data=data, headers=headers)
            if response.status_code < 300:
                individual_predictions.append(json.loads(response.text)['predictions'][0][0])
            else:
                individual_predictions.append(0)
        predictions.append(individual_predictions)
    return predictions


def run_osdg_models(abstracts):
    predictions = []
    for abstract in abstracts:
        data = {'text': abstract, 'detailed': True}
        response = requests.post('http://osdg:5000/tag', json=data)
        predictions.append(response.json())
    return predictions


def get_osdg_classifications(abstracts):
    results = run_osdg_models(abstracts=abstracts)
    predictions = []
    for result in results:
        prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for single_result in result['result']:
            sdg_number = single_result['sdg'].replace('SDG_', '')
            prediction[int(sdg_number) - 1] = 1
        predictions.append(prediction)
    return predictions


def run_mode_by_name(text, model):
    input_ids, masks = prepare_input(abstracts=text)
    input = {
        'input_ids': input_ids.numpy().tolist()[0],
        'attention_masks': masks.numpy().tolist()[0]
    }
    data = json.dumps({'instances': [input]})
    headers = {"content-type": "application/json"}
    serving_url = serving_base_url.format(model)
    url = f'{serving_url}/models/sdg{model}:predict'
    test = requests.get(f'{serving_url}/models/sdg{model}')
    app.logger.info(f'url_test for {url}: response = {test.text}')
    response = requests.post(url, data=data, headers=headers)
    app.logger.info(f'response for {url}: response = {response.text}')
    if response.status_code < 300:
        return json.loads(response.text)['predictions']
    else:
        return None


def run_model_by_name(term, model):
    if model == 'osdg':
        prediction = get_osdg_classifications(abstracts=[term])[0]
    elif model == 'elsevier-sdg-multi':
        prediction = run_mode_by_name(text=[term], model='elsevier')[0]
    elif model == 'aurora-sdg-multi':
        prediction = run_mode_by_name(text=[term], model='multi')[0]
    else:
        prediction = run_all_models(abstracts=[term])[0]
    return prediction


def get_prediction(text, model):
    prediction = run_model_by_name(text, model)
    response = []
    for index, sdg_value in enumerate(prediction):
        sdg_number = index + 1
        sdg_id = 'http://metadata.un.org/sdg/' + str(sdg_number)
        sdg_label = 'Goal ' + str(sdg_number)
        sdg_name = goal_names[sdg_label]
        icon = f'https://{classifier_host}/resources/sdg_icon_{sdg_number}.png'
        response.append(
            {"prediction": sdg_value,
             "sdg": {
                 "@type": "sdg",
                 "id": sdg_id,
                 "label": sdg_label,
                 "code": str(sdg_number),
                 "name": sdg_name,
                 "type": "Goal",
                 "icon": icon
             }})

    # create response object
    self_link = f'https://{classifier_host}/classify/{model}'
    bloated_result = {"@self": self_link,
                      "text": text,
                      "model": model,
                      "model_name": model_names[model],
                      "@context": {
                          "sdg": "http://metadata.un.org/sdg/",
                          "product": "https://schema.org/Product",
                          "model": "https://schema.org/Model",
                          "text": "https://schema.org/Text",
                          "icon": "https://schema.org/thumbnail",
                          "prediction": "https://schema.org/Float",
                          "xsd": "http://www.w3.org/2001/XMLSchema#"
                      },
                      "predictions": response}
    return bloated_result

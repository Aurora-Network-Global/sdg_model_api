import json

from flask import request

from app.service import model_service
from app.model import model_blueprint


@model_blueprint.route('/classify/<model>', methods=['POST'])
def run_model_of_type_by_post(model):
    term = request.get_json()['text']
    return json.dumps(model_service.get_prediction(term, model))


@model_blueprint.route('/classify/<model>', methods=['GET'])
def run_model_of_type_by_get(model):
    text = request.args.get('text')
    return json.dumps(model_service.get_prediction(text, model))

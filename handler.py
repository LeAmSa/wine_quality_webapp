#arquivo modelo para criação de uma API com FLASK para deploy
from flask import Flask, request
import os
import pandas as pd
import pickle
import json
from wine_quality.WineQuality import WineQuality

#carregar o modelo para mantê-lo na memória economizando tempo
model = pickle.load(open('model/model_wine_quality.pkl', 'rb')) #rb: modo leitura

#instanciando o flask
app = Flask(__name__)

#definindo os endpoints com as rotas
#cada endpoint executa sua função
@app.route('/predict', methods=['POST'])
def predict():
    #criando a variável que receberá os dados
    test_json = request.get_json()

    #coletando os dados
    if test_json:
        if isinstance(test_json, dict): #caso o valor seja único, leia como dicionário
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys()) #caso seja múltiplos valores, leia os valores das chaves json

    #instanciando a preparação dos dados
    pipeline = WineQuality()

    #preparação dos dados
    df1 = pipeline.data_preparation(df_raw)


    #com os dados recebidos, vamos realizar as predições e devolver ao usuário
    pred = model.predict(df1)
    df1['prediction'] = pred

    #retornando as predições em formato json
    return df1.to_json(orient='records')

#startando a api
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)


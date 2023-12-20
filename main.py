from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
scaler = MinMaxScaler()

def hitung_AQI_SO2(median_value):
    if 0.0 <= median_value < 35.0:
        return ((50.0 - 0.0) / (35.0 - 0.0)) * (median_value - 0.0) + 0.0
    elif 35.0 <= median_value < 75.0:
        return ((100.0 - 50.0) / (75.0 - 35.0)) * (median_value - 35.0) + 50.0
    elif 75.0 <= median_value < 185.0:
        return ((150.0 - 100.0) / (185.0 - 75.0)) * (median_value - 75.0) + 100.0
    elif 185.0 <= median_value < 304.0:
        return ((200.0 - 150.0) / (304.0 - 185.0)) * (median_value - 185.0) + 150.0
    elif 304.0 <= median_value < 604.0:
        return ((300.0 - 200.0) / (604.0 - 304.0)) * (median_value - 304.0) + 200.0
    elif 604.0 <= median_value < 804.0:
        return ((400.0 - 300.0) / (804.0 - 604.0)) * (median_value - 604.0) + 300.0
    else:
        return ((500.0 - 400.0) / (1004.0 - 805.0)) * (median_value - 805.0) + 400.0

def hitung_AQI_PM25(median_value):
    if 0.0 <= median_value < 12.0:
        return ((50.0 - 0.0) / (12.0 - 0.0)) * (median_value - 0.0) + 0.0
    elif 12.0 <= median_value < 35.4:
        return ((100.0 - 50.0) / (35.4 - 12.00)) * (median_value - 12.00) + 50.0
    elif 35.4 <= median_value < 55.4:
        return ((150.0 - 100.0) / (55.4 - 35.4)) * (median_value - 35.4) + 100.0
    elif 55.4 <= median_value < 150.4:
        return ((200.0 - 150.0) / (150.4 - 55.4)) * (median_value - 55.4) + 150.0
    elif 150.4 <= median_value < 250.4:
        return ((300.0 - 200.0) / (250.4 - 150.4)) * (median_value - 150.4) + 200.0
    elif 250.5 <= median_value < 350.4:
        return ((400.0 - 300.0) / (350.4 - 250.4)) * (median_value - 250.4) + 300.0
    else:
        return ((500.0 - 400.0) / (500.4 - 305.4)) * (median_value - 305.0) + 400.0

def hitung_AQI_PM10(median_value):
    if 0.0 <= median_value < 54.0:
        return ((50.0 - 0.0) / (54.0 - 0.0)) * (median_value - 0.0) + 0.0
    elif 54.0 <= median_value < 154.0:
        return ((100.0 - 50.0) / (154.0 - 54.00)) * (median_value - 54.00) + 50.0
    elif 154.0 <= median_value < 254.0:
        return ((150.0 - 100.0) / (254.0 - 154.0)) * (median_value - 154.0) + 100.0
    elif 254.0 <= median_value < 354.0:
        return ((200.0 - 150.0) / (354.0 - 254.0)) * (median_value - 254.0) + 150.0
    elif 354.0 <= median_value < 424.0:
        return ((300.0 - 200.0) / (424.0 - 354.0)) * (median_value - 354.0) + 200.0
    elif 424.0 <= median_value < 504.0:
        return ((400.0 - 300.0) / (504.0 - 424.0)) * (median_value - 424.0) + 300.0
    else:
        return ((500.0 - 400.0) / (604.0 - 504.0)) * (median_value - 504.0) + 400.00

def hitung_AQI_O3(median_value):
    if 0.0 <= median_value < 54.0:
        return ((50.0 - 0.0) / (54.0 - 0.0)) * (median_value - 0.0) + 0.0
    elif 54.0 <= median_value < 70.0:
        return ((100.0 - 50.0) / (70.0 - 54.00)) * (median_value - 54.00) + 50.0
    elif 70.0 <= median_value < 85.0:
        return ((150.0 - 100.0) / (85.0 - 70.0)) * (median_value - 70.0) + 100.0
    elif 85.0 <= median_value < 105.0:
        return ((200.0 - 150.0) / (105.0 - 85.0)) * (median_value - 85.0) + 150.0
    elif 105.0 <= median_value < 200.0:
        return ((300.0 - 200.0) / (200.0 - 105.0)) * (median_value - 105.0) + 200.0
    elif 200.0 <= median_value < 504.0:
        return ((400.0 - 300.0) / (504.0 - 200.0)) * (median_value - 200.0) + 300.0
    else:
        return ((500.0 - 400.0) / (604.0 - 504.0)) * (median_value - 504.0) + 400.0

def hitung_AQI_CO(median_value):
    if 0.0 <= median_value < 4.4:
        return ((50.0 - 0.0) / (4.4 - 0.0)) * (median_value - 0.0) + 0.0
    elif 4.4 <= median_value < 9.4:
        return ((100.0 - 50.0) / (9.4 - 4.40)) * (median_value - 4.40) + 50.0
    elif 9.4 <= median_value < 12.4:
        return ((150.0 - 100.0) / (12.4 - 9.4)) * (median_value - 9.4) + 100.0
    elif 12.4 <= median_value < 15.4:
        return ((200.0 - 150.0) / (15.4 - 12.4)) * (median_value - 12.4) + 150.0
    elif 15.4 <= median_value < 30.4:
        return ((300.0 - 200.0) / (30.4 - 15.4)) * (median_value - 15.4) + 200.0
    elif 30.4 <= median_value < 40.4:
        return ((400.0 - 300.0) / (40.4 - 30.4)) * (median_value - 30.4) + 300.0
    else:
        return ((500.0 - 400.0) / (50.4 - 40.4)) * (median_value - 40.4) + 400.0

def hitung_AQI_NO2(median_value):
    if 0.0 <= median_value < 54.0:
        return ((50.0 - 0.0) / (54.0 - 0.0)) * (median_value - 0.0) + 0.0
    elif 54.0 <= median_value < 100.0:
        return ((100.0 - 50.0) / (100.0 - 54.00)) * (median_value - 54.00) + 50.0
    elif 100.0 <= median_value < 360.0:
        return ((150.0 - 100.0) / (360.0 - 100.0)) * (median_value - 100.0) + 100.0
    elif 360.0 <= median_value < 649.0:
        return ((200.0 - 150.0) / (649.0 - 360.0)) * (median_value - 360.0) + 150.0
    elif 649.0 <= median_value < 1249.0:
        return ((300.0 - 200.0) / (1249.0 - 649.0)) * (median_value - 649.0) + 200.0
    elif 1249.0 <= median_value < 1649.0:
        return ((400.0 - 300.0) / (1649.0 - 1249.0)) * (median_value - 1249.0) + 300.0
    else:
        return ((500.0 - 400.0) / (2049.0 - 1649.0)) * (median_value - 1649.0) + 400.00

def determine_input(kota, data):
    output_features = []

    if kota == "jakarta":
        output_features = ["so2", "pm25", "pm10", "o3", "co", "no2"]
        model = load_model("model_jakarta.h5")

    elif kota == "semarang":
        output_features = ["pm25", "pm10"]
        model = load_model("model_semarang.h5")

    elif kota == "bandung":
        output_features = ["pm25", "pm10"]
        model = load_model("model_bandung.h5")

    input_data = np.array([[entry[features] for features in output_features] for entry in data["input_data"]])
    n_features_input = len(output_features)
    return model, input_data, n_features_input, output_features

def predict_per_24_hours(model, input_data, n_features_input, output_features):
    n_days = 3
    last_24_timesteps_scaled = input_data
    last_24_timesteps_scaled = scaler.fit_transform(last_24_timesteps_scaled)
    future_predictions = []

    for _ in range(24 * n_days):
        input_sequence = last_24_timesteps_scaled.reshape(1, 24, n_features_input)
        prediction = model.predict(input_sequence)
        prediction_scaled = scaler.inverse_transform(prediction)
        future_predictions.append(prediction_scaled.flatten())
        last_24_timesteps_scaled = np.concatenate((last_24_timesteps_scaled[1:], prediction), axis=0)

    return pd.DataFrame(future_predictions, columns=output_features)

def calculate_aqi(future_predictions_df, output_features):
    aqi_functions = {
        "so2": hitung_AQI_SO2,
        "pm25": hitung_AQI_PM25,
        "pm10": hitung_AQI_PM10,
        "o3": hitung_AQI_O3,
        "co": hitung_AQI_CO,
        "no2": hitung_AQI_NO2
    }

    for pollutant in output_features:
        if pollutant in aqi_functions and pollutant in future_predictions_df.columns:
            future_predictions_df[f"aqi_{pollutant}"] = future_predictions_df[pollutant].apply(aqi_functions[pollutant])

    aqi_days = []

    for day, daily_data in future_predictions_df.groupby(future_predictions_df.index // 24):
        daily_median_aqi = []

        for pollutant in output_features:
            if f"aqi_{pollutant}" in daily_data.columns:
                median_aqi = daily_data[f"aqi_{pollutant}"].median()
                daily_median_aqi.append({
                    "name": pollutant,
                    "aqi_median": median_aqi,
                })

        dominant_pollutant = max(daily_median_aqi, key=lambda x: x['aqi_median'])

        aqi_days.append({
            "dominant_pollutant": dominant_pollutant['name'],
            "dominant_pollutant_median": dominant_pollutant['aqi_median'],
            "pollutants": daily_median_aqi,
        })

    return future_predictions_df, aqi_days

def merge_values(future_predictions_df, aqi_days):

    predictions = {}

    for i, day_data in enumerate(aqi_days, start=1):
        per_hours_data = []

        for pollutant_data in day_data["pollutants"]:
            pollutant_name = pollutant_data["name"]
            aqi_values = future_predictions_df[f"aqi_{pollutant_name}"].tolist()
            concentration_values = future_predictions_df[pollutant_name].tolist()

            per_hours_data.append({
                "pollutant": pollutant_name,
                "data": {
                    "aqi": {hour: round(value) for hour, value in enumerate(aqi_values[i - 1:i + 23], start=1)},
                    "concentration": {hour: round(value, 2) for hour, value in
                                      enumerate(concentration_values[i - 1:i + 23], start=1)},
                }
            })

        per_hours_data = [
            {"data": entry["data"], "pollutant": entry["pollutant"]}
            for entry in per_hours_data
        ]

        predictions[f"next_{i}"] = {
            "medianAQI": round(day_data["dominant_pollutant_median"]),
            "dominantPollutant": day_data["dominant_pollutant"],
            "perHours": per_hours_data,
        }

    result = {"predictions": predictions}

    return result

def choose_model(kota):
    
    if kota == "jakarta":
        model = load_model("model_weather_jakarta.h5")

    elif kota == "semarang":
        model = load_model("model_weather_semarang.h5")

    elif kota == "bandung":
        model = load_model("model_weather_bandung.h5")

    return model
def konversi_variabel_laju(variabel):

    jumlah_data = len(variabel)
    laju = np.zeros(jumlah_data)

    for i in range(jumlah_data):
        if i == 0:
            laju[i] = variabel[i+1] - variabel[i]
        elif 0 < i < jumlah_data - 1:
            laju[i] = ((variabel[i] - variabel[i-1]) + (variabel[i+1] - variabel[i])) / 2
        elif i == jumlah_data - 1:
            laju[i] = variabel[i] - variabel[i-1]

    return laju


def input_preproces(input_data, features):
    output_data = {}

    for feature in features:
        output_data[feature] = []

    for entry in input_data:
        for feature in features:
            output_data[feature].extend(entry[feature])

    input_df = pd.DataFrame(output_data)

    for feature in features:
        rate_feature = feature + '_rate'
        input_df[rate_feature] = konversi_variabel_laju(input_df[feature])

    output_features = list(input_df.columns)

    input_scaled = scaler.fit_transform(input_df)

    return input_scaled, output_features

def predict_per_4_days(model, input_df, output_features):

    last_4_days = input_df
    future_predictions = []
    n_features_input = len(output_features)

    for _ in range(4):
        input_sequence = last_4_days.reshape(1, 4, n_features_input)
        prediction = model.predict(input_sequence)
        prediction_inversed = scaler.inverse_transform(prediction)
        future_predictions.append(prediction_inversed.flatten())
        last_4_days = np.concatenate((last_4_days[1:], prediction), axis=0)

    future_predictions_df = pd.DataFrame(future_predictions, columns=output_features)

    return future_predictions_df

def json_output(future_predictions_df, features):
    rounded_predictions_df = future_predictions_df[features].astype(float).round(2)
    rounded_predictions_df.index = rounded_predictions_df.index + 1
    json_result = {
        "predictions": rounded_predictions_df[features].to_dict(orient='index')
    }

    return json_result

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    try:
        data = request.get_json(force=True)
        kota = data['kota']
        input_data = data["input_data"]
        features = ['Tn', 'Tx', 'RH_avg', 'RR']

        model = choose_model(kota)

        input_scaled, output_features = input_preproces(input_data, features)

        future_predictions_df = predict_per_4_days(model, input_scaled, output_features)

        result = json_output(future_predictions_df, features)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict', methods=['POST'])
def predict_aqi():
    try:
        data = request.get_json(force=True)
        kota = data['kota']

        model, input_data, n_features_input, output_features = determine_input(kota, data)

        future_predictions_df = predict_per_24_hours(model, input_data, n_features_input, output_features)

        future_predictions_df, aqi_days = calculate_aqi(future_predictions_df, output_features)

        result = merge_values(future_predictions_df, aqi_days)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
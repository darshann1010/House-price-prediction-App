from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("linear_regression.pkl")

@app.route('/', methods=['GET'])
def predict():
    try:
        # Extract all params
        medinc = request.args.get('medinc')
        houseage = request.args.get('houseage')
        population = request.args.get('population')
        latitude = request.args.get('latitude')
        avgrms = request.args.get('avgrms')
        avgocc = request.args.get('avgocc')

        # DEBUGGING: Print what was received
        print("medinc:", medinc)
        print("houseage:", houseage)
        print("population:", population)
        print("latitude:", latitude)
        print("avgrms:", avgrms)
        print("avgocc:", avgocc)

        # Check if any value is missing
        if None in [medinc, houseage, population, latitude, avgrms, avgocc]:
            return jsonify({'error': 'One or more parameters are missing'}), 400

        # Convert to float
        input_data = np.array([[float(medinc), float(houseage), float(population),
                                float(avgocc), float(avgrms), float(latitude)]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction * 100000, 2)

        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

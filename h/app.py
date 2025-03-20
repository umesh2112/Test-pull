import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model, encoder & feature columns
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
feature_columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
                # Collect form data
                        input_data = {
                                    'District': [request.form["district"]],
                                                'Crop': [request.form["crop"]],
                                                            'Market': [request.form["market"]],
                                                                        'Nitrogen': [float(request.form["N"])],
                                                                                    'Phosphorus': [float(request.form["P"])],
                                                                                                'Potassium': [float(request.form["K"])],
                                                                                                            'Temperature': [float(request.form["temperature"])],
                                                                                                                        'Rainfall': [float(request.form["rainfall"])],
                                                                                                                                    'Humidity': [float(request.form["humidity"])],
                                                                                                                                                'pH_Value': [float(request.form["pH"])],
                                                                                                                                                            'Year': [int(request.form["year"])],
                                                                                                                                                                        'Month': [int(request.form["month"])],
                                                                                                                                                                                    'Day': [int(request.form["day"])]
                                                                                                                                                                                            }

                                                                                                                                                                                                    df = pd.DataFrame(input_data)

                                                                                                                                                                                                            # Encode categorical features
                                                                                                                                                                                                                    encoded_data = pd.DataFrame(encoder.transform(df[['District', 'Crop', 'Market']]))
                                                                                                                                                                                                                            encoded_data.columns = encoder.get_feature_names_out(['District', 'Crop', 'Market'])
                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                            # Merge numeric and encoded data
                                                                                                                                                                                                                                                    df = df.drop(['District', 'Crop', 'Market'], axis=1)
                                                                                                                                                                                                                                                            df = pd.concat([df, encoded_data], axis=1)

                                                                                                                                                                                                                                                                    # Align columns with training data
                                                                                                                                                                                                                                                                            df = df.reindex(columns=feature_columns, fill_value=0)

                                                                                                                                                                                                                                                                                    # Predict crop price
                                                                                                                                                                                                                                                                                            predicted_price = model.predict(df)[0]

                                                                                                                                                                                                                                                                                                    return render_template("result.html", prediction=round(predicted_price, 2))

                                                                                                                                                                                                                                                                                                        except Exception as e:
                                                                                                                                                                                                                                                                                                                return f"Error: {str(e)}"

                                                                                                                                                                                                                                                                                                                if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                    app.run(debug=True)
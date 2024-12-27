from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
app.template_folder = "C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/templates"
app.debug = True

# Load model linear regression dari disk
with open("C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/models/linear_regression.pkl", "rb") as linear_regression_file:
    linear_regression_model = pickle.load(linear_regression_file)

# Load model MLP (regressor) dari disk
with open("C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/models/mlp_regressor.pkl", "rb") as mlp_regressor_file:
    mlp_regressor_model = pickle.load(mlp_regressor_file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_price", methods=["POST"])
def predict_price():
    house_age = request.form["house_age"]
    dist_to_MRT = request.form["dist_to_MRT"]
    conv_store = request.form["conv_store"]

    # Validasi input
    try:
        house_age = float(house_age)
        dist_to_MRT = float(dist_to_MRT)
        conv_store = float(conv_store)
    except ValueError:
        return jsonify({"error": "Input tidak valid."})

    # Lakukan prediksi dengan model linear regression
    linear_regression_prediction = linear_regression_model.predict([[house_age, dist_to_MRT, conv_store]])

    # Lakukan prediksi dengan model MLP
    mlp_regressor_prediction = mlp_regressor_model.predict([[house_age, dist_to_MRT, conv_store]])

    # Render template dengan hasil prediksi dari kedua model
    return render_template("index.html", 
                           linear_regression_prediction=linear_regression_prediction[0],
                           mlp_regressor_prediction=mlp_regressor_prediction[0])

if __name__ == "__main__":
    app.run()

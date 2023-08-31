from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pandas as pd
import joblib

# 1. Create the Application Object
PropertyPricePredictionApp = Flask(__name__)

# 2. Load the model from disk
fileName = 'property_price_prediction_model.sav'
loaded_model = joblib.load(fileName)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@PropertyPricePredictionApp.route('/')
def index():
    heading = 'Property Price Prediction Model'
    return render_template("home.html", heading=heading)


@PropertyPricePredictionApp.route('/home.html')
def back_to_index():
    heading = 'Property Price Prediction Model'
    return render_template("home.html", heading=heading)


@PropertyPricePredictionApp.route('/describeProject.html')
@cross_origin()
def home():
    heading = 'Property Price Prediction'
    return render_template("describeProject.html", heading=heading)


# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted price with the confidence (http://127.0.0.1:8000/predictprice)

@PropertyPricePredictionApp.route("/predictor.html", methods=["GET", "POST"])
@cross_origin()
def price_prediction():
    if request.method == "POST":
        Property_Type = float(request.form["PropertyType"])
        Club_House = int(request.form['Club House'])
        School_University_In_Township = int(request.form['SchoolUniversityInTownship'])
        Hospital_In_Township = int(request.form['Hospital_In_Township'])
        Mall_In_Township = int(request.form['Mall_In_Township'])
        Park_Jogging_Track = int(request.form['Park_Jogging_Track'])
        Swimming_Pool = int(request.form['Swimming_Pool'])
        Gym = int(request.form["Gym"])
        Property_Area_in_SqFt = float(request.form['Property Area in SqFt'])
        Price_By_SubArea = float(request.form['Price_By_SubArea'])
        Amenities_Score = int(request.form['Amenities_Score'])
        Price_By_Amenities_Score = float(request.form['Price_By_Amenities_Score'])
        Noun_Counts = 0
        Verb_Counts = 0
        Adjective_Counts = 0
        boasts_elegant = 0
        elegant_towers = 0
        every_day = 0
        great_community = 0
        mantra_gold = 0
        offering_bedroom = 0
        quality_specification = 0
        stories_offering = 0
        towers_stories = 0
        world_class = 0

        data_df = [[Property_Type,
                    Club_House,
                    School_University_In_Township,
                    Hospital_In_Township,
                    Mall_In_Township,
                    Park_Jogging_Track,
                    Swimming_Pool,
                    Gym,
                    Property_Area_in_SqFt,
                    Price_By_SubArea,
                    Amenities_Score,
                    Price_By_Amenities_Score,
                    Noun_Counts,
                    Verb_Counts,
                    Adjective_Counts,
                    boasts_elegant,
                    elegant_towers,
                    every_day,
                    great_community,
                    mantra_gold,
                    offering_bedroom,
                    quality_specification,
                    stories_offering,
                    towers_stories,
                    world_class]]
        print(data_df)
        predicted_value = loaded_model.predict(data_df)
        print(str(predicted_value))
        return render_template('predictor.html', prediction_text="Property price is Rs. {}".format(predicted_value))
    return render_template("predictor.html")


@PropertyPricePredictionApp.route("/retrain.html", methods=['GET', 'POST'])
def model_training():
    status = ' '
    if request.method == "POST":
        model_training_flag = float(request.form["RetrainModel"])
        if model_training_flag == 1:
            status = "Model Training Completed"
        else:
            status = "Model Training was selected as No"

    return render_template("retrain.html", model_training_status=status)


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    PropertyPricePredictionApp.run(debug=True)
    PropertyPricePredictionApp.config['TEMPLATES_AUTO_RELOAD'] = True

# End-to-End-Real-Estate-Price-Prediction-Model
This is a sample project for price prediction of Real Estate Property. Model is trained on dataset with Property Price as the Target Variable. The model takes into account the important factors such as Sub-Area, Propert Type, Property Area in Sq. Ft, Price in lakhs, Price in Millions and Total TownShip Area in Acres and other features as ClubHouse, School/University in Township, Hospital in TownShip, Mall in TownShip, Park/Jogging track,	Swimming Pool and Gym.

## Installation
```
$ git clone https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model.git
$ cd End-to-End-Real-Estate-Price-Prediction-Model
$ python -m venv venv/
$ source venv/Scripts/activate
$ pip install -r requirements.txt
```

## Tech Stack:
 - **Language**: Python
 - **Libraries**: Flask, gunicorn, scipy, linear regression, Ridge Regression, Lasso Regression, joblib, seaborn, scikit_learn, NLTK
 - **Services:** Flask, Deployment on Heroku Platform

## Project Approach:
 - Create the repository in github in local
 - **Application Code**
    - Flask app (app.py) is created various modules\features on the webpage as given below:
        - Home Page - It briefly describes the Project
        - Project Description - This provides the techniques used for the modelling
        - Price Prediction - It will accept the features from webpage for single property and predict the price of the property.
        - Model Training - If we have a different set of data on which we want to train our model, we can copy the files in the input folder in bit bucket and Train the model from the webpage.
    - ModelTrainingEngine module will serve the model training on new data from the webpage.
    - model_Pipeline module is used for reading the data, data preprocessing, data encoding and outlier treatment.


 - **Model Deployment**
    - runtime.txt - is used to create python images for model deployment
    - Procfile - is used for running the Flask on Heroku Platform.
    - requirements.txt - is used to install library for deploying the models. 
    - nltk.txt - is used to install various nltk modules for NLP related tasks.


- Exploratory Data Analysis:

  * ***[No of Property in each sub Area](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/subArea_propertyCount.png)***
    
  ![Images No of Property in each sub Area](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/subArea_propertyCount.png)


  * ***[HousesBuildbyEachCountry](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/HousesBuildbyEachCountry.png)***
    
  ![Images HousesBuildbyEachCountry](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/HousesBuildbyEachCountry.png)


- Linear Regression, Lasso Regression and Ridge Regression Algorithm are used and model is evaluated for performance based on RMSE , MAE as performance metrics.
- Screenshot for Inference Pipeline and Model Training
  * ***[Home Page](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/home-page.jpg)***
    
  ![Image Home Page](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/home-page.jpg)


  * ***[Price Prediction](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/model-prediction.jpg)***
    
  ![Image Price Prediction](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/model-prediction.jpg)


  * ***[Model Training](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/model-training.jpg)***
    
  ![Model Training Image](https://github.com/sudhanshusinghaiml/End-to-End-Real-Estate-Price-Prediction-Model/blob/develop/images/model-training.jpg)
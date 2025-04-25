import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix  
import matplotlib.pyplot as plt  

# here we are loading the data and training the Model
# we chose random forest because it was giving the best results for accuracy.
# here we are loading the patient data from a CSV file 
df = pd.read_csv("diabetestest.csv")

# here we are making the prediciton fetures that will match our dataset colums
features = [
    'high_bp',                  
    'highChol',                 
    'BMI',                      
    'smoker',                 
    'stroke',                
    'physicalactivity',         
    'hvyalcohol_consumption',   
    'Sex'                       
]

# here we hae made a target as 0 or 1. if patient is predicted diabetic or not
target = 'diabetes_binary'

# here we are setting x as input and y as outputs
X = df[features]
y = df[target]


# Random Forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# here we are training our model on the dtaset we provided
rf_model.fit(X, y)

y_pred = rf_model.predict(X)

# here we are checking the accuracy on how often we got the correct prediction
accuracy = accuracy_score(y, y_pred)

#below is confusion matrix which will tell us how many are wrongly predicted
conf_matrix = confusion_matrix(y, y_pred)



# below is the function for inputs, when we get an iput it will compare here
def predict_diabetes(input_data):
    
    # here we will use our trained model to make the prediction
    prediction = rf_model.predict(input_data)[0]
    return prediction

# here we are showing which inputs we have priortized, like high bp.
def get_feature_importance():

    importances = rf_model.feature_importances_
    feature_names = X.columns
    return feature_names, importances

#here we are making a graph to show what the model actually finds imprtant

# below is a bar chart to represent that
def plot_feature_importance():
   
    feature_names, importances = get_feature_importance()

    plt.figure(figsize=(10, 6))  
    plt.barh(feature_names, importances, color="#0B5394")  
    plt.xlabel("Feature Importance")  
    plt.ylabel("Features")  
    plt.title("Random Forest Feature Importance")  
    plt.tight_layout()  
    plt.show()

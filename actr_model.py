import pandas as pd

# here we are making a  ACT-R Cognitive Model D


class ACTRModel:
    def __init__(self):  
        # here we are making a declarative memory to store patient data
        self.declarative_memory = {}
    # here we are putting chunks of patient datta in memory
    def declare_chunk(self, chunk_id, **attributes):
        self.declarative_memory[chunk_id] = attributes 
    
    # here we are searching and finding all the patients that will match the descriptin
    def retrieve_all_similar_patients(self, query):
        matching_patients = [] # here we are creating a list to store all the mathes

        # here we are going through every patient in memory
        for chunk_id, attributes in self.declarative_memory.items():
            match = True  

            # here we are using a for loop to check if the patients info match the query
            for key, value in query.items():
                if attributes.get(key) != value:
                    match = False  
                    break

            # If we get a match, we add them to our list
            if match:
                matching_patients.append({'chunk_id': chunk_id, **attributes})

        return matching_patients

# here we are loading patients from dataset

def load_patients_to_actr(model, csv_file):
    
    df = pd.read_csv(csv_file)

    df = df.head(60000)

    # classifying BMI into global doctor approved groups
    def bmi_group(bmi):
        if bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        else:
            return 'obese'

    # here we are applying that bmi grouping to each patient
    df['bmi_group'] = df['BMI'].apply(bmi_group)

    # using for loop for each patient and store their data in ACTR memory
    for idx, row in df.iterrows():
        model.declare_chunk(
            str(idx),  
            high_bp='yes' if row['high_bp'] == 1 else 'no',
            highChol='yes' if row['highChol'] == 1 else 'no',
            bmi_group=row['bmi_group'],
            smoker='yes' if row['smoker'] == 1 else 'no',
            stroke='yes' if row['stroke'] == 1 else 'no',
            phys_activity='yes' if row['physicalactivity'] == 1 else 'no',
            alcohol='yes' if row['hvyalcohol_consumption'] == 1 else 'no',
            sex='male' if row['Sex'] == 1 else 'female',
            diagnosis='yes' if row['diabetes_binary'] == 1 else 'no'
        )

# down we are going to predict with ACTR Based on Query 
# this function tales in all the patient info and predicts if the patient is likely diabetic or not
def predict_diabetes_actr(model, query):
   
    # here we are finding similar patients from memory
    matching_patients = model.retrieve_all_similar_patients(query)

    diabetic_count = 0
    non_diabetic_count = 0

    # making a for loop to find diabetic and non diabetic patients
    for patient in matching_patients:
      if patient['diagnosis'] == 'yes':
         diabetic_count=diabetic_count+ 1  
      else:
          non_diabetic_count = non_diabetic_count+ 1
    total_patients = diabetic_count +  non_diabetic_count
    # If no patients matched the query, we can't really decide
    if total_patients == 0:
        return "uncertain", 0.0, True  

    # finally making the Diagnosis 

    # here we are calculating the percentage of diabetic patients
    total_patients = diabetic_count + non_diabetic_count
    if total_patients > 0:
        diabetic_percentage = (diabetic_count / total_patients) * 100
    else:
        diabetic_percentage = 0  


    #  if the patients are too less to compare with, we are not confident and will send a warning
    if total_patients < 20:
        print("\nWarning: Very few matching patients. The diagnosis might not be reliable.")


    #we have decided to set the threshold to 60%
    threshold = 60

    # finally deciding if they could be diabitic or not
    if diabetic_percentage >= threshold:
        actr_diagnosis = 'mostly diabetic'
    elif diabetic_percentage <= (100 - threshold):
        actr_diagnosis = 'mostly non-diabetic'
    else:
        actr_diagnosis = 'uncertain'
    warning_flag = total_patients < 20
    # Return the decision, percentage, and whether there was a warning
    return actr_diagnosis, diabetic_percentage, warning_flag


actr_model = ACTRModel()

load_patients_to_actr(actr_model, 'diabetestest.csv')

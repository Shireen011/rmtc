# ID: Patients Identifier
# WBC: White Blood Cell, Normal Ranges: 4.0 to 10.0, Unit: 10^9/L.
# LYMp: Lymphocytes percentage, which is a type of white blood cell, Normal Ranges: 20.0 to 40.0, Unit: %
# MIDp: Indicates the percentage combined value of the other types of white blood cells not classified as lymphocytes or granulocytes, Normal Ranges: 1.0 to 15.0, Unit: %
# NEUTp: Neutrophils are a type of white blood cell (leukocytes); neutrophils percentage, Normal Ranges: 50.0 to 70.0, Unit: %
# LYMn: Lymphocytes number are a type of white blood cell, Normal Ranges: 0.6 to 4.1, Unit: 10^9/L.
# MIDn: Indicates the combined number of other white blood cells not classified as lymphocytes or granulocytes, Normal Ranges: 0.1 to 1.8, Unit: 10^9/L.
# NEUTn: Neutrophils Number, Normal Ranges: 2.0 to 7.8, Unit: 10^9/L.
# RBC: Red Blood Cell, Normal Ranges: 3.50 to 5.50, Unit: 10^12/L
# HGB: Hemoglobin, Normal Ranges: 11.0 to 16.0, Unit: g/dL
# HCT: Hematocrit is the proportion, by volume, of the Blood that consists of red blood cells, Normal Ranges: 36.0 to 48.0, Unit: %
# MCV: Mean Corpuscular Volume, Normal Ranges: 80.0 to 99.0, Unit: fL
# MCH: Mean Corpuscular Hemoglobin is the average amount of haemoglobin in the average red cell, Normal Ranges: 26.0 to 32.0, Unit: pg
# MCHC: Mean Corpuscular Hemoglobin Concentration, Normal Ranges: 32.0 to 36.0, Unit: g/dL
# RDWSD: Red Blood Cell Distribution Width, Normal Ranges: 37.0 to 54.0, Unit: fL
# RDWCV: Red blood cell distribution width, Normal Ranges: 11.5 to 14.5, Unit: %
# PLT: Platelet Count, Normal Ranges: 100 to 400, Unit: 10^9/L
# MPV: Mean Platelet Volume, Normal Ranges: 7.4 to 10.4, Unit: fL
# PDW: Red Cell Distribution Width, Normal Ranges: 10.0 to 17.0, Unit: %
# PCT: The level of Procalcitonin in the Blood, Normal Ranges: 0.10 to 0.28, Unit: %
# PLCR: Platelet Large Cell Ratio, Normal Ranges: 13.0 to 43.0, Unit: %


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, confusion_matrix

# Load the dataset
data = pd.read_csv('bloodReport.csv')

# Features (X):
# Select relevant features for clustering
# Platelet count (PLT), Hemoglobin (HGB), Hematocrit (HCT), White Blood Cell count (WBC)
X = data[['PLT', 'HGB', 'HCT', 'WBC']]

# Apply K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Get cluster assignments for each patient
cluster_assignments = kmeans.labels_

# Assign disease categories to the clusters based on patterns (you can adjust this based on domain knowledge)
def assign_disease_category(cluster_label):
    if cluster_label == 0:
        return 'Dengue'
    elif cluster_label == 1:
        return 'Malaria'
    elif cluster_label == 2:
        return 'Scrub Thypus'
    elif cluster_label == 3:
        return 'Leptospirosis'
    else:
        return 'Unknown'

# Add the 'DiseaseCategory' column to the original DataFrame
data['DiseaseCategory'] = [assign_disease_category(label) for label in cluster_assignments]

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier using all data
rf_classifier.fit(X, data['DiseaseCategory'])

# Make predictions on the data for disease category
predicted_disease_category = rf_classifier.predict(X)

# Evaluate the model
accuracy = accuracy_score(data['DiseaseCategory'], predicted_disease_category)
f1 = f1_score(data['DiseaseCategory'], predicted_disease_category, average='weighted')
precision = precision_score(data['DiseaseCategory'], predicted_disease_category, average='weighted')
conf_matrix = confusion_matrix(data['DiseaseCategory'], predicted_disease_category)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision Score:", precision)
print("\nConfusion Matrix:\n", conf_matrix)

# Classification Report
print("\nClassification Report:")
print(classification_report(data['DiseaseCategory'], predicted_disease_category))

# Save the updated DataFrame to a CSV file
data.to_csv('classifiedReport.csv', index=False)

# Print the patients' disease categories
print("Patients Categorized into Disease Groups:")
print(data[['ID', 'DiseaseCategory']])

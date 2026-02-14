import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("c:first/cs_students.csv")
print(df.head())
print(df.columns)
# STEP 1: Load Dataset
# =====================
data = pd.read_csv("c:first/cs_students.csv")

print("\nDataset Loaded Successfully!\n")
print(data.head())

# ==================================
# STEP 2: Encode Categorical Columns
#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        encoders[col]=le
        data[col] = le.fit_transform(data[col])
    # Save encoder for decoding later

print("\nCategorical Columns Encoded!\n")

# =====================
# STEP 3: REGRESSION
# GPA Prediction
# =====================
print("----- REGRESSION: GPA Prediction -----")

X_reg = data[['Age', 'Python', 'SQL', 'Java', 'Projects']]
y_reg = data['GPA']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

gpa_predictions = reg_model.predict(X_test_reg)

print("Regression MAE (GPA Error):", mean_absolute_error(y_test_reg, gpa_predictions))

# ============================
# STEP 4: CLASSIFICATION
# Future Career Prediction
# ============================
print("\n----- CLASSIFICATION: Career Prediction -----")

X_clf = data[['GPA', 'Python', 'SQL', 'Java', 'Major', 'Interested Domain']]
y_clf = data['Future Career']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf, y_train_clf)

career_predictions = clf_model.predict(X_test_clf)

print("Classification Accuracy:", accuracy_score(y_test_clf, career_predictions))


# =====================
# STEP 5: New Student Prediction (Optional)
# =====================
a = int(input("Enter your age: "))
b = float(input("Please enter your GPA: "))

new_student = pd.DataFrame({
    'Age': [a],                # user input use karo
    'Python': [8],
    'SQL': [7],
    'Java': [6],
    'Projects': [3],
    'Major': [1],
    'Interested Domain': [2],
    'GPA': [b]                 # user input use karo
})


predicted_gpa = reg_model.predict(new_student[['Age', 'Python', 'SQL', 'Java', 'Projects']])
predicted_career = clf_model.predict(new_student[['GPA', 'Python', 'SQL', 'Java', 'Major', 'Interested Domain']])
decoded_career = encoders['Future Career'].inverse_transform(predicted_career)

print("Predicted Career (Name):", decoded_career[0])

print("Predicted GPA:", predicted_gpa[0])
print("Predicted Career (Encoded):", predicted_career[0])

print("\nProject Execution Completed Successfully!")


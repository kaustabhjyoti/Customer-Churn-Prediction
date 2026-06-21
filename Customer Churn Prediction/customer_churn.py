import pandas as pd   

df = pd.read_csv("customer_churn.csv")

print(df.head())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["Churn"] = encoder.fit_transform(df["Churn"])

print(df.head())

X = df.drop("Churn", axis=1)

y = df["Churn"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data: ", len(X_train))
print("Testing Data: ", len(X_test))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model Successfully Trained!")

predictions = model.predict(X_test)

print(predictions)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: ", accuracy)


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
print(feature_importance)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

import seaborn as sns  
import matplotlib.pyplot as plt 

correlation = df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Customer Churn Correlation Heatmap")

plt.savefig("Churn_heatmap.png")
plt.show()
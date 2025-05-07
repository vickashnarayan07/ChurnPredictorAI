import pandas as pd
import tkinter as tk
from tkinter import messagebox, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initial Dataset
data = {
    'Age': [
        25, 35, 45, 23, 52, 46, 51, 44, 28, 39, 29, 41, 33, 38, 50, 48, 55, 36, 30, 47,
        21, 58, 32, 40, 49, 27, 53, 31, 37, 43, 24, 42, 34, 56, 26, 60, 22, 54, 57, 59
    ],
    'Gender': [
        'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female',
        'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
        'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
        'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male'
    ],
    'Contract': [
        'Month-to-month', 'Two year', 'One year', 'Month-to-month', 'One year', 'Month-to-month',
        'Two year', 'One year', 'Month-to-month', 'Two year',
        'Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year',
        'Month-to-month', 'One year', 'Two year', 'Month-to-month',
        'One year', 'Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Two year',
        'One year', 'Month-to-month', 'Two year', 'Month-to-month',
        'One year', 'Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Two year',
        'Month-to-month', 'One year', 'Two year', 'Month-to-month'
    ],
    'MonthlyCharges': [
        70, 80, 60, 90, 40, 75, 85, 60, 95, 55,
        65, 70, 50, 90, 45, 85, 95, 60, 80, 75,
        55, 95, 65, 70, 85, 90, 50, 75, 60, 55,
        85, 70, 95, 50, 90, 45, 95, 65, 55, 85
    ],
    'Churn': [
        'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No',
        'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes',
        'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes',
        'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'
    ]
}

df = pd.DataFrame(data)

# Preprocessing
le_gender = LabelEncoder()
le_contract = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Contract'] = le_contract.fit_transform(df['Contract'])

X = df[['Age', 'Gender', 'Contract', 'MonthlyCharges']]
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression()

# Train Function
def train_model():
    output_text.delete('1.0', tk.END)  # clear output
    full_report = ""

    for i in range(1, 4):
        full_report += f"--- Training Round {i} ---\n"
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        full_report += f"Accuracy: {accuracy*100:.2f}%\n"
        full_report += f"{report}\n\n"

    output_text.insert(tk.END, full_report)
    messagebox.showinfo("Training Complete", "Model trained successfully!")

# Predict Function
def predict_customer():
    try:
        age = int(age_entry.get())
        gender_text = gender_var.get()
        contract_text = contract_var.get()
        charges = float(charges_entry.get())

        gender_encoded = le_gender.transform([gender_text])[0]
        contract_encoded = le_contract.transform([contract_text])[0]

        new_customer = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Contract': [contract_encoded],
            'MonthlyCharges': [charges]
        })

        new_customer_scaled = scaler.transform(new_customer)
        prediction = model.predict(new_customer_scaled)

        result = "Yes" if prediction[0] == 1 else "No"
        messagebox.showinfo("Prediction", f"Churn Prediction for New Customer: {result}")

    except Exception as e:
        messagebox.showerror("Input Error", f"Please enter valid data.\n\nError: {str(e)}")

# GUI
window = tk.Tk()
window.title("Simple AI Project: Churn Prediction")
window.geometry("700x700")
window.config(bg="#e0f7fa")

# Input Fields
tk.Label(window, text="Enter New Customer Details", font=("Arial", 16), bg="#e0f7fa").pack(pady=10)

form_frame = tk.Frame(window, bg="#e0f7fa")
form_frame.pack(pady=10)

tk.Label(form_frame, text="Age:", font=("Arial", 12), bg="#e0f7fa").grid(row=0, column=0, padx=10, pady=5)
age_entry = tk.Entry(form_frame, font=("Arial", 12))
age_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(form_frame, text="Gender:", font=("Arial", 12), bg="#e0f7fa").grid(row=1, column=0, padx=10, pady=5)
gender_var = tk.StringVar(value="Male")
gender_menu = tk.OptionMenu(form_frame, gender_var, "Male", "Female")
gender_menu.config(font=("Arial", 12))
gender_menu.grid(row=1, column=1, padx=10, pady=5)

tk.Label(form_frame, text="Contract:", font=("Arial", 12), bg="#e0f7fa").grid(row=2, column=0, padx=10, pady=5)
contract_var = tk.StringVar(value="Month-to-month")
contract_menu = tk.OptionMenu(form_frame, contract_var, "Month-to-month", "One year", "Two year")
contract_menu.config(font=("Arial", 12))
contract_menu.grid(row=2, column=1, padx=10, pady=5)

tk.Label(form_frame, text="Monthly Charges:", font=("Arial", 12), bg="#e0f7fa").grid(row=3, column=0, padx=10, pady=5)
charges_entry = tk.Entry(form_frame, font=("Arial", 12))
charges_entry.grid(row=3, column=1, padx=10, pady=5)

# Buttons
train_button = tk.Button(window, text="Train Model", command=train_model, font=("Arial", 14), bg="#4CAF50", fg="white", width=20)
train_button.pack(pady=20)

predict_button = tk.Button(window, text="Predict New Customer", command=predict_customer, font=("Arial", 14), bg="#2196F3", fg="white", width=20)
predict_button.pack(pady=10)

# Output Area
output_text = scrolledtext.ScrolledText(window, width=80, height=20, font=("Courier", 10))
output_text.pack(pady=20)

# Start GUI
window.mainloop()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv('IA/Proyecto_Diagnosticos/sintomas_medicos.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model, 'logistic_model.pkl')
model = joblib.load('logistic_model.pkl')
symptoms = X.columns.tolist()
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90) 
plt.show()

# Crear la ventana principal
root = tk.Tk()
root.title("Sistema de Diagnóstico Médico")

# Lista para almacenar las intensidades de los síntomas
intensity_vars = []

# Función para hacer la predicción
def predict_disease():
    input_data = np.array([int(var.get()) for var in intensity_vars]).reshape(1, -1)
    prediction = model.predict(input_data)
    result_label.config(text=f"Diagnóstico: {prediction[0]}")

# Crear los widgets para los síntomas
for symptom in symptoms:
    frame = ttk.Frame(root)
    frame.pack(fill='x', padx=5, pady=5)
    label = ttk.Label(frame, text=symptom.replace('_', ' '))
    label.pack(side='left', padx=5)
    intensity_var = tk.IntVar(value=0)
    combobox = ttk.Combobox(frame, textvariable=intensity_var, values=[0, 1, 2, 3, 4, 5], width=5)
    combobox.pack(side='right', padx=5)
    intensity_vars.append(intensity_var)

# Botón para hacer la predicción
predict_button = ttk.Button(root, text="Diagnosticar", command=predict_disease)
predict_button.pack(pady=10)

# Label para mostrar el resultado
result_label = ttk.Label(root, text="Diagnóstico: ")
result_label.pack(pady=5)

# Iniciar el bucle principal de la interfaz
root.mainloop()

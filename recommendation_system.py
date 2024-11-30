# recommendation_system.py

# Importar las librerias necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos (ejemplo: dataset de productos)
# Este dataset debe estar previamente descargado o puede ser generado
# Aquí se crea un dataset ficticio para el ejemplo
data = {
    'producto_id': [1, 2, 3, 4, 5],
    'caracteristica_1': [10, 20, 10, 30, 20],
    'caracteristica_2': [1, 0, 1, 0, 1],
    'etiqueta': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Preprocesamiento de datos
# Seleccionar las caracteristicas y la etiqueta
X = df[['caracteristica_1', 'caracteristica_2']]
y = df['etiqueta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Funcion de recomendacion
def recomendar(producto_id, modelo, datos):
    producto = datos[datos['producto_id'] == producto_id]
    if producto.empty:
        return "Producto no encontrado"
    caracteristicas = producto[['caracteristica_1', 'caracteristica_2']]
    prediccion = modelo.predict(caracteristicas)
    return "Recomendado" if prediccion[0] == 1 else "No recomendado"

# Ejemplo de uso de la funcion de recomendacion
producto_id = 3
recomendacion = recomendar(producto_id, model, df)
print(f'Recomendación para el producto {producto_id}: {recomendacion}')
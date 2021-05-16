import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
 
import requests
# Tratamiento de datos
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
 
# Gráficos
# ------------------------------------------------------------------------------

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
 
# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
 
 
#Se guarda el código de la página
documento = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
valores = documento.content
lin = valores.decode('utf-8')
lineas = lin.split('\n')
valores=[]
columnas=["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
 
for b in range (len(lineas)):
  v = lineas[b].split(',')
  lineas[b] = v
  
 
X=[]
Y=[]
lineasBorradas=[]
 
# Se quitan las lineas con caracteres invalidos

for i in range (len(lineas)-1):
  a=[]
  for e in range (len(lineas[i])-1):
    a.append(lineas[i][e])  
  if (a[6]) != ("?"): 
    X.append(a)
  else:
    lineasBorradas.append(i)

#Mete en Y los valores que debe predecir el modelo. Los valores deben contar con lineas validas.

for i in range (len(lineas)-1):
  for n in range (len(lineas[i])-1, len(lineas[i])):
    if i not in lineasBorradas:
      Y.append(lineas[i][n])
 

modelo = DecisionTreeRegressor()

 
# Entrenamiento del modelo
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

modelo.fit(X_train, y_train)
 
# Estructura del árbol creado
# ------------------------------------------------------------------------------


#X_train=pd.DataFrame(X,columns=columnas)

print(f"Profundidad del árbol: {modelo.get_depth()}")
print(f"Número de nodos terminales: {modelo.get_n_leaves()}")
 
plot = plot_tree(
            decision_tree = modelo,
            feature_names = columnas,
            filled        = True,
            impurity      = False,
            fontsize      = 10,
            precision     = 2,
            #ax            = ax
       )

#Prediccion del modelo

y_pred = modelo.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicho':y_pred})

#Mostrar modelo
df

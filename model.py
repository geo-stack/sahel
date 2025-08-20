#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# In[2]:


df_1 = pd.read_csv("Topo_features_Mali_500_30_4_3.csv", index_col=0)

for file_path in os.listdir():
    if "Topo_features_Mali_500_30" in file_path and file_path!="Topo_features_Mali_500_30_4_3.csv":
        df_2 = pd.read_csv(file_path, index_col=0)
        df_1 = pd.concat([df_1, df_2])

df_1


# In[4]:


df_topo = df_1.drop(["CODE_OUVRA", "LON", "LAT", "DATE", "NS", "precipitation", "ndvi"], axis=1).dropna(axis=0) #Attention: la colonne "CODE_OUVRA est pour le Mali ici
                                                                                                                #Pour les autres pays cette colonne peut changer de nom


# In[5]:


df_meteo = pd.read_csv("Meteo_features_Mali.csv", index_col=0)

df = pd.concat([df_meteo, df_topo], axis=1).dropna(axis=0)


# In[6]:


df


# In[7]:


import folium

# Centre de la carte (moyenne des coordonnées)
center_lat, center_lon = df["LAT"].mean(), df["LON"].mean()

# Créer la carte
m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# Ajouter les points
for _, row in df.iterrows():
    color = "red" if row["LON"] < -6.2 else "blue"  # Délimitation
    folium.CircleMarker(
        location=[row["LAT"], row["LON"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
    ).add_to(m)

# Ajouter une ligne pour délimiter -6.2
folium.PolyLine(
    locations=[[df["LAT"].min(), -6.2], [df["LAT"].max(), -6.2]],
    color="black",
    weight=2,
    opacity=0.7
).add_to(m)

# Afficher la carte
m


# In[10]:


train_index = df[df["LON"]<-6.2].index
test_index = df[df["LON"]>=-6.2].index

print(len(train_index), len(test_index))


# In[ ]:


X_topo = df.drop([
    "CODE_OUVRA",
    "LON",
    "LAT",
    "DATE",
    "NS",
    "ridge_row",
    "ridge_col",
    "stream_row",
    "stream_col",
], axis=1)
y = df["NS"]

X_train = X_topo.loc[train_index]
y_train = y.loc[train_index]
X_test = X_topo.loc[test_index]
y_test = y.loc[test_index]

model = RandomForestRegressor(n_estimators=50, random_state=42)

#Entrainement du modèle à partir de toutes les features (résultats pas terribles)
model.fit(X_train, y_train)


# In[13]:


#On visualise ici les résultats sur le dataset de test de notre modèle entraîné

y_pred = model.predict(X_test)

print(f"RMSE (en %): {round(np.sqrt(np.mean((y_test - y_pred)**2))/np.mean(y_test) * 100, 3)} %")
print(f"Corrélation entre les prédictions et les vraies valeurs: {np.corrcoef(y_test, y_pred)[0, 1].round(3)*100} %")

max_limit = max(20, 20)

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred)

T = np.arange(0, max_limit + 1)
plt.plot(T, T, 'r')

plt.xlim(0, max_limit)
plt.ylim(0, max_limit)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('True vs Predicted values')

plt.show()


# In[14]:


#On recommence mais en testant itérativement plusieurs combinaisons (impossible de tout tester car cela prendrait trop de temps)
#Donc on arrête de faire tourner la cellule au bout d'un moment ~20 minutes par exemple

results = []

for i in range(6, len(X_topo.columns) + 1): #6 représente ici la longueur des combinaisons de features à tester. On peut tester plusieurs longueurs différentes
    for j, subset in enumerate(combinations(X_topo.columns, i)):
        if j%100==0:
            print(j)
        X_subset = X_topo[list(subset)]

        X_train = X_subset.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X_subset.loc[test_index]
        y_test = y.loc[test_index]  
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(np.mean((y_test - y_pred)**2)) / np.mean(y_test) * 100
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        results.append({
            "features": subset,
            "rmse": round(rmse, 3),
            "correlation": round(correlation, 3)
        })

results_df = pd.DataFrame(results)

best_rmse = results_df.sort_values(by="rmse", ascending=True).head(10)
print("Top 10 combinaisons avec les meilleurs RMSE :")
print(best_rmse)

best_corr = results_df.sort_values(by="correlation", ascending=False).head(10)
print("\nTop 10 combinaisons avec les meilleures corrélations :")
print(best_corr)


# In[15]:


#On affiche les résultats par ordre décroissant de résultats

results_df = pd.DataFrame(results)

best_rmse = results_df.sort_values(by="rmse", ascending=True).head(10)
print("Top 10 combinaisons avec les meilleurs RMSE :")
display(best_rmse)

best_corr = results_df.sort_values(by="correlation", ascending=False).head(10)
print("\nTop 10 combinaisons avec les meilleures corrélations :")
print(best_corr)


# In[49]:


best_rmse.iloc[0]["features"]


# In[ ]:


X = df[[
'ndvi',
 'precipitation',
 'area',
 'ratio_alt',
 'ridge_grad_skew',
 'short_grad_mean',
]]
y = df["NS"]

X_train = X.loc[train_index]
y_train = y.loc[train_index]
X_test = X.loc[test_index]
y_test = y.loc[test_index]

model = RandomForestRegressor(n_estimators=50, random_state=42)

model.fit(X_train, y_train)


# In[51]:


y_pred = model.predict(X_test)

print(f"RMSE (en %): {round(np.sqrt(np.mean((y_test - y_pred)**2))/np.mean(y_test) * 100, 3)} %")
print(f"Corrélation entre les prédictions et les vraies valeurs: {np.corrcoef(y_test, y_pred)[0, 1].round(3)*100} %")


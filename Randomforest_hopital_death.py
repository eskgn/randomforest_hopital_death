
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"

path = "C:/Users/enisk/Desktop/Programmation/DataChallenge2/dataPT2.csv"
pd.set_option('display.max_columns', None)
df = pd.read_csv(path, header=0, sep=";")


df['bmi'] = df['bmi'].str.replace(',','.')
df['bmi'] = df['bmi'].str.replace(r'[^\d.]', '', regex=True)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['ethnicity'] = df['ethnicity'].replace('Other/Unknown', np.nan)
df['height'] = df['height'].str.replace(',', '.').astype(float)
#icu_admit_source
df['pre_icu_los_days'] = df['pre_icu_los_days'].str.replace(',', '.').astype(float)
df['pre_icu_los_days'] = pd.to_numeric(df['pre_icu_los_days'], errors='coerce')
df['pre_icu_los_days'] = df['pre_icu_los_days'].apply(lambda x: x if x >= 0 else None)
df['apache_3j_diagnosis'] = df['apache_3j_diagnosis'].str.replace(',', '.').astype(float)
# df['apache_3j_diagnosis'] = pd.to_numeric(df['apache_3j_diagnosis'], errors='coerce')
df['weight'] = df['weight'].str.replace(',','.')
df['weight'] = df['weight'].str.replace(r'[^\d.]', '', regex=True)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['resprate_apache'] = df['resprate_apache'].str.replace(',','.').astype(float)
df['temp_apache'] = df['temp_apache'].str.replace(',','.').astype(float)
df['d1_sysbp_noninvasive_min'] = df['d1_sysbp_noninvasive_min'].str.replace(',', '.').astype(float)
df['d1_temp_max'] = df['d1_temp_max'].str.replace(',', '.').astype(float)
df['d1_temp_min'] = df['d1_temp_min'].str.replace(',', '.').astype(float)
df['d1_potassium_max'] = df['d1_potassium_max'].str.replace(',', '.').astype(float)
df['d1_potassium_min'] = df['d1_potassium_min'].str.replace(',', '.').astype(float)
df['apache_4a_hospital_death_prob'] = df['apache_4a_hospital_death_prob'].str.replace(',', '.').astype(float)
df['apache_4a_icu_death_prob'] = df['apache_4a_icu_death_prob'].str.replace(',', '.').astype(float)
df['apache_2_bodysystem'] = df['apache_2_bodysystem'].replace('undefined Diagnoses', 'Undefined Diagnoses')
# apache_4a_hospital_death_prob : 2118 valeur = -1 ?
df['apache_4a_hospital_death_prob'] = df['apache_4a_hospital_death_prob'].replace(-1,np.nan)
df['gender'] = df['gender'].replace('F', 0)
df['gender'] = df['gender'].replace('M', 1)

#VALEUR 0 ? 0 = souvent mort, une très petite proportion survie donc à garder
# df['d1_spo2_max'] = df['d1_spo2_max'].replace(0, np.nan)
# df['d1_resprate_min'] = df['d1_resprate_min'].replace(0, np.nan)
# df = df[(df['d1_heartrate_min'] >= 20)]
# df = df[(df['d1_spo2_min'] >= 20)]


#####################################################################################################################
#Création des colonnes binaires à partir des scores  0 : valeur normal 1 : anormal
#####################################################################################################################
# gcs_motor_apache : 1,2,3 -> 1 et 4,5,6 -> 0
df['gcs_motor_apache'] = df['gcs_motor_apache'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# gcs_eyes_apache : 1,2,3 -> 1 et 4 -> 0
df['gcs_eyes_apache'] = df['gcs_eyes_apache'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# gcs_verbal_apache : 1,2,3,4 -> 1 et 5 -> 0
df['gcs_verbal_apache'] = df['gcs_verbal_apache'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

#S'il y a au moins 40% de chance de mourir avec le diagnostic alors il est recodé 1
recodage = ['apache_2_diagnosis','apache_3j_diagnosis','d1_glucose_min']
for var in recodage:
    df[var] = np.where(df[var].isin(df.groupby(var)['hospital_death'].mean().loc[lambda x: x >= 0.3].index), 0, 1)

df['heart_rate_apache'] = df['heart_rate_apache'].apply(lambda x: 0 if 34 < x < 140 else 1)

df['map_apache'] = df['map_apache'].apply(lambda x: 0 if 48 < x else 1)
df['temp_apache'] = df['temp_apache'].apply(lambda x: 1 if 35.5 < x < 39.5 else 0)
df['d1_diasbp_min'] = df['d1_diasbp_min'].apply(lambda x: 0 if 33 < x else 1)
df['d1_heartrate_max'] = df['d1_heartrate_max'].apply(lambda x: 1 if 134 < x else 0)
df['d1_heartrate_min'] = df['d1_heartrate_min'].apply(lambda x: 0 if 28 < x else 1)
df['d1_mbp_min'] = df['d1_mbp_min'].apply(lambda x: 0 if 40 < x else 1)
df['d1_mbp_noninvasive_min'] = df['d1_mbp_noninvasive_min'].apply(lambda x: 0 if 40 < x else 1)
df['d1_spo2_min'] = df['d1_spo2_min'].apply(lambda x: 0 if 75 < x else 1)
df['d1_sysbp_min'] = df['d1_sysbp_min'].apply(lambda x: 0 if 67 < x else 1)
df['d1_sysbp_noninvasive_min'] = df['d1_sysbp_noninvasive_min'].apply(lambda x: 0 if 66 < x else 1)
df['d1_temp_max'] = df['d1_temp_max'].apply(lambda x: 0 if 35.72 < x < 39.78 else 1)
df['d1_temp_min'] = df['d1_temp_min'].apply(lambda x: 0 if 33.8 < x else 1)
df['h1_diasbp_min'] = df['h1_diasbp_min'].apply(lambda x: 0 if 31 < x else 1)
df['h1_diasbp_noninvasive_min'] = df['h1_diasbp_noninvasive_min'].apply(lambda x: 0 if 33 < x else 1)
df['h1_mbp_min'] = df['h1_mbp_min'].apply(lambda x: 0 if 43 < x else 1)
df['h1_mbp_noninvasive_min'] = df['h1_mbp_noninvasive_min'].apply(lambda x: 0 if 45 < x else 1)
df['h1_spo2_min'] = df['h1_spo2_min'].apply(lambda x: 1 if 10 < x < 75 else 0)
df['h1_sysbp_max'] = df['h1_sysbp_max'].apply(lambda x: 0 if 79 < x else 1)
df['h1_sysbp_min'] = df['h1_sysbp_min'].apply(lambda x: 0 if 60 < x else 1)
df['h1_sysbp_noninvasive_max'] = df['h1_sysbp_noninvasive_max'].apply(lambda x: 0 if 95 < x else 1) #séparation non précise
df['h1_sysbp_noninvasive_min'] = df['h1_sysbp_noninvasive_min'].apply(lambda x: 0 if 68 < x else 1)
df['d1_glucose_max'] = df['d1_glucose_max'].apply(lambda x: 1 if 508 < x < 600 else 0)


variable_qualitative_categorielle = [
    'apache_3j_bodysystem',
    'apache_2_diagnosis',
    'apache_3j_diagnosis',
    'apache_2_bodysystem',
    'icu_stay_type',
]
#Variables qualitatives binaires (conserver)
variable_qualitative_binaire = [
    'arf_apache',
    'intubated_apache',
    'ventilated_apache',
    'heart_rate_apache',
    'aids',
    'cirrhosis',
    'diabetes_mellitus',
    'hepatic_failure',
    'immunosuppression',
    'leukemia',
    'solid_tumor_with_metastasis',
    'gcs_unable_apache',
    'lymphoma',
    'gcs_eyes_apache',
    'gcs_motor_apache',
    'gcs_verbal_apache',
    'apache_post_operative',
    'hospital_death'
]

#Variables quantitatives (conserver)
variable_quantitative = [
    'age',
    'pre_icu_los_days',
    'map_apache',
    'resprate_apache',
    'temp_apache',
    'd1_glucose_max',
    'd1_glucose_min',
    'd1_potassium_max',
    'd1_potassium_min',
    'd1_spo2_min',
    'd1_heartrate_max',
    'd1_heartrate_min',
    'weight',
    'd1_diasbp_min',
    'd1_mbp_noninvasive_min',
    'd1_mbp_min',
    'd1_sysbp_min',
    'd1_sysbp_noninvasive_min',
    'd1_temp_max',
    'd1_temp_min',
    'h1_diasbp_min',
    'h1_diasbp_noninvasive_min',
    'h1_heartrate_max',
    'h1_heartrate_min',
    'h1_mbp_min',
    'h1_mbp_noninvasive_min',
    'h1_spo2_min',
    'h1_sysbp_max',
    'h1_sysbp_min',
    'h1_sysbp_noninvasive_max',
    'h1_sysbp_noninvasive_min',
]

#Variables retirées
variable_id = [
    #Identifiants
    'encounter_id',
    'patient_id',
    'hospital_id',
    'icu_id',
    
    #Issues de variable_qualitative_categorielle
    'icu_type',
    'ethnicity',
    'icu_admit_source',
    
    #Issues de variable_qualitative_binaire
    'gender',
    'elective_surgery',
    
    #Issues de variable_quantitative
    'apache_4a_hospital_death_prob', # probabilité calculé
    'apache_4a_icu_death_prob', # probabilité calculé
    'height',
    'd1_diasbp_max',
    'd1_mbp_noninvasive_max',
    'd1_diasbp_noninvasive_min',
    'd1_diasbp_noninvasive_max',
    'd1_sysbp_max',
    'd1_sysbp_noninvasive_max',
    'h1_diasbp_max',
    'h1_diasbp_noninvasive_max',
    'h1_mbp_max',
    'h1_mbp_noninvasive_max',
    'h1_spo2_max',
    'd1_spo2_max',
    'd1_mbp_max',
    'd1_resprate_max',
    'd1_resprate_min',
    'h1_resprate_max',
    'h1_resprate_min'
]

df = df.drop(columns=variable_id)
for col in variable_qualitative_categorielle:
    df[col] = df[col].astype(str)
for col in variable_qualitative_binaire:
    df[col] = df[col].astype(str)

df = df.replace('nan', np.nan)

# ##########retirer les valeurs aberrantes
# for var in variable_quantitative:
#     # Calcul des quantiles 1er et 3e quartiles
#     Q1 = df[var].quantile(0.25)
#     Q3 = df[var].quantile(0.75)
    
#     # Calcul de l'IQR (Interquartile Range)
#     IQR = Q3 - Q1
    
#     # Définition des bornes pour les valeurs aberrantes
#     lower_bound = Q1 - 5 * IQR
#     upper_bound = Q3 + 5 * IQR
    
#     # Filtrer les valeurs aberrantes en utilisant les bornes
#     df = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]


#########################################
# Analyse descriptive et visualisation
#########################################
# cat_col = []
# num_col = []
#
# for col in df.columns[3:]:
#     print(col.upper())
#     print("\n", df[col].head(5), "\n")
    
#     try:
#         print(f"Description: {dict_description[col]}")
#     except (NameError, KeyError):
#         print("Description: No description available")
        
#     print("\nNumber of Null values:")
#     print(df[col].isnull().sum())
    
#     print("\nValue Counts:")
#     print(df[col].value_counts())
#     print("\n", df[col].describe(include="all"), "\n")
  
#     if df[col].nunique() > 20:
#         plt.figure(figsize=(12,8))
#         # Affichage de la distribution en séparant les cas de survie et de décès
#         sns.histplot(df[df['hospital_death']=='0'][col].dropna(), color='g', label='Survive', kde=True, edgecolor='black')
#         sns.histplot(df[df['hospital_death']=='1'][col].dropna(), color='r', label='Death', kde=True, edgecolor='black')
#         plt.legend()
#         plt.title(col)
#         plt.show()        
#         num_col.append(col)       
#     else:
#         plt.figure(figsize=(14, 6))
#         sns.countplot(x=col, hue="hospital_death", data=df, palette='coolwarm')
#         plt.legend(loc='upper right')
#         plt.yscale('log')
#         plt.xticks(rotation=45)
#         plt.title(col)
#         plt.show()
        
#         if df[col].dtype != object:
#             num_col.append(col)
#         else:
#             cat_col.append(col)     
#     print('_______________________________________________________________________________')


    

print(df['hospital_death'].value_counts())

#Charger les données 
X = df.drop(columns=["hospital_death"]) 
y = df["hospital_death"]

variable_qualitative_binaire.remove('hospital_death')

################################################################################################
# Séparation des données en ensembles d'entraînement et de test avant prépocessing
################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

################################################################################################
# preprocessing
################################################################################################

# proportion_nan = df.isna().mean()
# print(proportion_nan)

#Pour données entrainement
print('Remplacement des données manquantes...')
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(strategy="median")  #Remplace les NaN par la médiane
X_train[variable_quantitative] = imputer_num.fit_transform(X_train[variable_quantitative])
imputer_cat = SimpleImputer(strategy="most_frequent")
X_train[variable_qualitative_categorielle] = imputer_cat.fit_transform(X_train[variable_qualitative_categorielle])
imputer_cat = SimpleImputer(strategy="most_frequent")
X_train[variable_qualitative_binaire] = imputer_cat.fit_transform(X_train[variable_qualitative_binaire])

#Pour données test
print('Remplacement des données manquantes...')
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(strategy="median")  #Remplace les NaN par la médiane
X_test[variable_quantitative] = imputer_num.fit_transform(X_test[variable_quantitative])
imputer_cat = SimpleImputer(strategy="most_frequent")
X_test[variable_qualitative_categorielle] = imputer_cat.fit_transform(X_test[variable_qualitative_categorielle])
imputer_cat = SimpleImputer(strategy="most_frequent")
X_test[variable_qualitative_binaire] = imputer_cat.fit_transform(X_test[variable_qualitative_binaire])


#scaler = StandardScaler()
#df[variable_quantitative] = scaler.fit_transform(df[variable_quantitative])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

for col in variable_qualitative_binaire:
    X_train[col] = X_train[col].astype(float).astype(int)
    X_test[col] = X_test[col].astype(float).astype(int)
y_train = y_train.astype(float).astype(int)
y_test = y_test.astype(float).astype(int)

# encoder = LabelEncoder()
# for col in variable_qualitative_binaire:
#     df[col] = encoder.fit_transform(df[col])

#Séparation pour l'encodage des données categorielle

#Pour données entrainement
X_train_hors_categorielle = X_train[variable_quantitative + variable_qualitative_binaire]
X_train_categorielle = X_train[variable_qualitative_categorielle]
      
X_train_categorielle_encoded  = pd.get_dummies(X_train_categorielle, columns=variable_qualitative_categorielle, drop_first=True).astype(int)
X_train = pd.concat([X_train_hors_categorielle, X_train_categorielle_encoded], axis=1)

#Pour données test
X_test_hors_categorielle = X_test[variable_quantitative + variable_qualitative_binaire]
X_test_categorielle = X_test[variable_qualitative_categorielle]
      
X_test_categorielle_encoded  = pd.get_dummies(X_test_categorielle, columns=variable_qualitative_categorielle, drop_first=True).astype(int)
X_test = pd.concat([X_test_hors_categorielle, X_test_categorielle_encoded], axis=1)

# print((df.isna().sum() / len(df)) * 100) #Vérification des NaN


#########################################################################
#Undersampling ou Oversampling
#########################################################################

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=0.29)
X_train_smote, y_train_smote = undersample.fit_resample(X_train, y_train)

# from imblearn.over_sampling import ADASYN
# adasyn = ADASYN(sampling_strategy=0.5, n_neighbors=2, random_state=42)
# X_train_smote, y_train_smote = adasyn.fit_resample(X_train, y_train)

# from imblearn.combine import SMOTETomek
# smote_tomek = SMOTETomek(sampling_strategy=0.3, random_state=42)
# X_train_smote, y_train_smote = smote_tomek.fit_resample(X_train, y_train)

# from imblearn.combine import SMOTEENN
# smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
# X_train_smote, y_train_smote = smote_enn.fit_resample(X_train, y_train)


print(y_train.count())
print("Distribution avec :", Counter(y_train_smote))
print("Distribution y_test :", Counter(y_test))

################################################################################################
#Entrainement
################################################################################################

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=35,
    max_features='sqrt',
    min_samples_split=20,
    min_samples_leaf=2,
    bootstrap = True,
    random_state=42,
    class_weight={0: 1, 1: 1.2},
    n_jobs = -1
)

#Validation croisée stratifiée avec StratifiedKFold
cv = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
f1_scores = []
seuil = 0.48  #À ajuster
#Boucle sur chaque fold de validation croisée
for train_index, val_index in cv.split(X_train_smote, y_train_smote):
    #Séparation des données en train et validation
    X_train_fold, X_val_fold = X_train_smote.iloc[train_index], X_train_smote.iloc[val_index]
    y_train_fold, y_val_fold = y_train_smote.iloc[train_index], y_train_smote.iloc[val_index]
    
    #Entraînement du modèle sur le jeu d'entraînement du fold
    rf.fit(X_train_fold, y_train_fold)
    
    #Prédictions sur le jeu de validation
    y_proba_fold = rf.predict_proba(X_val_fold)[:, 1]
    
    #Application du seuil pour classer les patients
    y_pred_fold = (y_proba_fold >= seuil).astype(int)
    
    #Calcul du F1 score pour ce fold
    f1_scores.append(f1_score(y_val_fold, y_pred_fold))
    
mean_f1_score = np.mean(f1_scores)
print(f"F1-score moyen sur validation croisée : {mean_f1_score:.3f}")

#Entraînement
rf.fit(X_train_smote, y_train_smote)

#Prédictions
y_proba = rf.predict_proba(X_test)[:, 1]
#Appliquer le seuil pour classer les patients
y_pred = (y_proba >= seuil).astype(int)


################################################################################################
#Évaluation du modèle
################################################################################################

f1_scores = f1_score(y_test, y_pred)
print(f"f1 test du modèle : {f1_scores:.3f}")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.3f}")

#Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.title("Matrice de confusion")
plt.show()

################################################################################################
# Affichage de l'importance des caractéristiques
################################################################################################
importances = rf.feature_importances_

#Créer un DataFrame pour l'affichage
feature_names = X_train_smote.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

#Trier les caractéristiques par importance décroissante
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df.head(5))

#Affichage graphique de l'importance des caractéristiques
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:60], importance_df['Importance'][:60])
plt.xlabel('Importance')
plt.title('30 des caractéristiques les plus importantes')
plt.show()

#AUC Score
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.3f}")

#Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

#Tracé de la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  #Ligne 0.5
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC')
plt.legend()
plt.grid()
plt.show()

#Courbe Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, marker='.', label="Precision-Recall Curve", color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall")
plt.legend()
plt.grid()
plt.show()

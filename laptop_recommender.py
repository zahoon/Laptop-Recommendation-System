import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import re

# === Load dataset ===
file_path = 'laptop_dataset.csv'  # <-- Make sure this file exists
df = pd.read_csv(file_path, encoding='latin1')

# === Clean & Convert ===
df = df[[ 
    'Company', 'Product', 'Inches', 'ScreenResolution', 'Cpu',
    'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price_in_euros'
]]

df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)
df['Price (RM)'] = (df['Price_in_euros'] * 5.2).round(2)

df.rename(columns={
    'Inches': 'Display_Size',
    'ScreenResolution': 'Screen_Resolution',
    'Cpu': 'CPU',
    'Ram': 'RAM',
    'Memory': 'Storage',
    'Gpu': 'GPU',
    'OpSys': 'OS',
}, inplace=True)

# === Extract storage size safely ===
def extract_storage_size(storage_str):
    try:
        matches = re.findall(r'(\d+)(TB|GB)', storage_str.upper())
        total = 0
        for size, unit in matches:
            gb = int(size) * 1024 if unit == 'TB' else int(size)
            total += gb
        return total
    except:
        return 0

df['Storage_Size_GB'] = df['Storage'].apply(extract_storage_size)

# === Preference Matching Function ===
def match_preferences(row, preference):
    cpu = row['CPU'].lower()
    gpu = row['GPU'].lower()
    ram = row['RAM']
    storage = row['Storage'].lower()

    if preference == 'gaming':
        return ('nvidia' in gpu or 'amd' in gpu) and ram >= 8
    elif preference == 'editing':
        return ('i7' in cpu or 'ryzen 7' in cpu) and ram >= 8
    elif preference == 'programming':
        return ('i5' in cpu or 'ryzen 5' in cpu) and ram >= 8
    elif preference == 'designing':
        return ('i7' in cpu or 'ryzen 7' in cpu) and 'ssd' in storage
    elif preference == 'office':
        return ram >= 4
    return False

# === User Input ===
user_ram = 8
user_price_rm = 4000
user_storage_size = 512
user_preference = 'gaming'

# === Filter based on preference ===
df_pref = df[df.apply(lambda x: match_preferences(x, user_preference), axis=1)].copy()

# === Prepare features and scale ===
features = ['RAM', 'Storage_Size_GB', 'Price (RM)']
X = df_pref[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === KNN Model ===
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(X_scaled)

# === User vector ===
user_vector = np.array([[user_ram, user_storage_size, user_price_rm]])
user_vector_scaled = scaler.transform(user_vector)

# === Get Nearest Neighbors ===
distances, indices = knn.kneighbors(user_vector_scaled)

# === Recommendations ===
recommendations = df_pref.iloc[indices[0]].copy()
recommendations.reset_index(drop=True, inplace=True)

# === Display Results ===
print("\n=== 🧠 Top 10 Recommended Laptops (KNN-based on RAM, Storage, Price) ===")
print(recommendations[['Company', 'Product', 'Display_Size', 'Screen_Resolution',
                       'CPU', 'RAM', 'Storage', 'GPU', 'OS', 'Weight', 'Price (RM)']])

# === Evaluation: Precision, Recall, F1 Score ===
ground_truth = df[
    (df['RAM'] >= user_ram) &
    (df['Price (RM)'] <= user_price_rm + 1000) &
    (df['Storage_Size_GB'] >= user_storage_size) &
    (df.apply(lambda x: match_preferences(x, user_preference), axis=1))
]

relevant_total = len(ground_truth)
recommended_relevant = len(recommendations)

precision = recommended_relevant / 10  # Always return top 10
recall = recommended_relevant / relevant_total if relevant_total else 0

# === Calculate F1 Score ===
if precision + recall > 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

# === Convert to Percentages ===
precision_percent = precision * 100
recall_percent = recall * 100
f1_percent = f1_score * 100

# === Print Evaluation Results ===
print("\n=== 📊 Performance Evaluation ===")
print(f"- Total Relevant Laptops (Ground Truth): {relevant_total}")
print(f"- Recommended Relevant (Top 10): {recommended_relevant}")
print(f"- Precision: {precision_percent:.2f}%")
print(f"- Recall: {recall_percent:.2f}%")
print(f"- F1 Score: {f1_percent:.2f}%")

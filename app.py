from flask import Flask, render_template, request, session
import math
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.secret_key = 'laptop'

# === Load and Clean Dataset ===
df = pd.read_csv('laptop_dataset.csv', encoding='latin1')
df = df[['Company', 'Product', 'Inches', 'ScreenResolution', 'Cpu',
         'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price_in_euros']]

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

# === Extract storage in GB ===
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

# === Match user preference ===
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

# === Routes ===
@app.route('/laptops')
def laptops():
    page = int(request.args.get('page', 1))
    per_page = 20  # Number of laptops per page
    df = pd.read_csv('laptop_dataset.csv', encoding='latin1')
    total = len(df)
    pages = math.ceil(total / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    laptops = df.iloc[start:end]
    return render_template('laptop.html', laptops=laptops, page=page, pages=pages)
@app.route('/evaluation')
def evaluation_page():
    evaluation = session.get('evaluation', None)
    return render_template('evaluation.html', evaluation=evaluation)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # User input
    ram = int(request.form.get('ram'))
    price = float(request.form.get('price'))
    storage_size = float(request.form.get('storage'))
    preference = request.form.get('preference').lower()

    # Filter by preference
    df_pref = df[df.apply(lambda x: match_preferences(x, preference), axis=1)].copy()
    if df_pref.empty:
        return render_template('index.html', message="No laptops match your preference.")

    # Prepare KNN
    features = ['RAM', 'Storage_Size_GB', 'Price (RM)']
    df_pref.dropna(subset=features, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pref[features])
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(X_scaled)

    user_vector = np.array([[ram, storage_size, price]])
    user_scaled = scaler.transform(user_vector)

    distances, indices = knn.kneighbors(user_scaled)
    recommendations = df_pref.iloc[indices[0]].copy()

    # Evaluation
    ground_truth = df[
        (df['RAM'] >= ram) &
        (df['Price (RM)'] <= price + 1000) &
        (df['Storage_Size_GB'] >= storage_size) &
        (df.apply(lambda x: match_preferences(x, preference), axis=1))
    ]
    relevant_total = len(ground_truth)
    recommended_relevant = len(recommendations)

    precision = recommended_relevant / 10
    recall = recommended_relevant / relevant_total if relevant_total else 0

    evaluation = (
        "=== Performance Evaluation ===<br>"
        f"Total Relevant Laptops (Ground Truth): {relevant_total}<br>"
        f"Recommended Relevant (Top 10): {recommended_relevant}<br>"
        f"Precision: {precision:.2f}<br>"
        f"Recall: {recall:.2f}"
    )
    session['evaluation'] = evaluation

    # Output tables split into 2 tabs
    table1 = recommendations.iloc[:5][[
        'Company', 'Product', 'Display_Size', 'Screen_Resolution',
        'CPU', 'RAM', 'Storage', 'GPU', 'OS', 'Weight', 'Price (RM)'
    ]].to_html(classes='table table-bordered', index=False)

    table2 = recommendations.iloc[5:][[
        'Company', 'Product', 'Display_Size', 'Screen_Resolution',
        'CPU', 'RAM', 'Storage', 'GPU', 'OS', 'Weight', 'Price (RM)'
    ]].to_html(classes='table table-bordered', index=False)

    user_inputs = {
        'RAM': ram,
        'Price': price,
        'Storage': storage_size,
        'Preference': preference.capitalize()
    }

    return render_template('index.html', tables=[table1, table2], evaluation=evaluation, user_inputs=user_inputs)

if __name__ == '__main__':
    app.run(debug=True)

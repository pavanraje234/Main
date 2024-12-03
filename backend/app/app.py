from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import os
import  csv
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from flask_cors import CORS
import json
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import traceback
import google.generativeai as genai
import io
import base64
from flask import after_this_request
import aivisualization
import GraphGenerator
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
DATABASE = 'db/my_database.db'
def extract_csv(pathname: str) -> list[dict]:
    """Extracts the content of the CSV into a list of dictionaries with headers as keys."""
    data = []
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)  
        for row in csv_reader:
            data.append(row)
    return data

# Configure Google Generative AI
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC-u1iHOoklku1mvfOZtW9Umr0UWF8tkDU")
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Database helper function
def query_db(query, args=(), one=False):
    con = sqlite3.connect(DATABASE)
    cur = con.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    con.close()
    return (rv[0] if rv else None) if one else rv

# Helper function to check for key terms in the question
def check_for_precise_answer(question: str):
    """Returns a precise answer if the question matches predefined ones."""
    question_lower = question.lower()
    for key, value in cards_data.items():
        if key in question_lower:
            return value
    return None


def query_db(query, args=(), one=False):
    con = sqlite3.connect(DATABASE)
    cur = con.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    con.close()
    return (rv[0] if rv else None) if one else rv



# API routes for sales and profitability data (remains unchanged)
@app.route('/api/sales_by_channel')
def sales_by_channel():
    query = """
        SELECT source, SUM(ordered_quantity) as total_sales
        FROM Orders_New_query_2024_07_01
        GROUP BY source
    """
    data = query_db(query)
    result = [{'channel': row[0], 'sales': row[1]} for row in data]
    return jsonify(result)

@app.route('/api/sales_by_sku')
def sales_by_sku():
    query = """
        SELECT sku_id, SUM(ordered_quantity) as total_sales
        FROM Orders_New_query_2024_07_01
        GROUP BY sku_id
    """
    data = query_db(query)
    result = [{'sku': row[0], 'sales': row[1]} for row in data]
    return jsonify(result)

@app.route('/api/sku_profitability')
def sku_profitability():
    query = """
        SELECT o.sku_id, 
               SUM(o.gross_merchandise_value - (o.ordered_quantity * c.unit_price)) as profitability
        FROM Orders_New_query_2024_07_01 o
        JOIN calculated_cogs_2024_07_01 c ON o.order_id = c.order_id
        GROUP BY o.sku_id
    """
    data = query_db(query)
    result = [{'sku': row[0], 'profitability': row[1]} for row in data]
    return jsonify(result)

@app.route('/api/profitability_by_channel')
def profitability_by_channel():
    query = """
        SELECT o.source, 
               SUM(o.gross_merchandise_value - (o.ordered_quantity * c.unit_price)) as profitability
        FROM Orders_New_query_2024_07_01 o
        JOIN calculated_cogs_2024_07_01 c ON o.order_id = c.order_id
        GROUP BY o.source
    """
    data = query_db(query)
    result = [{'channel': row[0], 'profitability': row[1]} for row in data]
    return jsonify(result)

# Chatbot route using Google Generative AI
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_question = request.json.get('question')

    modified_question = f"{user_question} this is just a comment - If the question is related to the dataset, provide a concise and precise answer in a full sentence without including code or extra information. Otherwise, respond normally, as you would in a casual conversation."
    # Extract CSV data    # Extract CSV data
    csv_data = aivisualization.extract_csv("Orders_New_query_2024_07_01.csv")

    # Start a chat session with the model
    chat_session = aivisualization.model.start_chat(history=[{
        "role": "user",
        
        "parts": [{"text": str(csv_data)}]
    }])

    # Send the user's question to the model
    response = chat_session.send_message(modified_question)
    
    # Extract the response
    answer = response.text.split("**Output:**")[1].split("**")[0] if "**Output:**" in response.text else response.text

    return jsonify({'answer': answer.strip()})

##Recommendation

# Load your dataset
data = pd.read_csv('Orders_New_query_2024_07_01.csv')

def recommend_items_with_similarity(area, item, data, n_recommendations=5):
    data['billing_address_state'] = data['billing_address_state'].str.strip().str.upper()
    area_item_matrix = data.pivot_table(index='billing_address_state', columns='sku_id', values='ordered_quantity', aggfunc='sum').fillna(0)
    item_similarity_matrix = pd.DataFrame(cosine_similarity(area_item_matrix.T), index=area_item_matrix.columns, columns=area_item_matrix.columns)
    area = area.strip().upper()

    if area not in area_item_matrix.index:
        return None

    items_in_area = area_item_matrix.loc[area].sort_values(ascending=False)

    if item not in items_in_area or items_in_area[item] == 0:
        return items_in_area.head(n_recommendations)

    recommendations = items_in_area[items_in_area.index != item].head(n_recommendations)
    similarity_scores = item_similarity_matrix.loc[item, recommendations.index]

    recommendations_with_similarity = pd.DataFrame({
        'SKU ID': recommendations.index,
        'Ordered Quantity': recommendations.values,
        'Gross Merchandise Value': data.groupby('sku_id')['gross_merchandise_value'].sum().reindex(recommendations.index, fill_value=0).values,
    }).sort_values(by='Ordered Quantity', ascending=False)

    return recommendations_with_similarity[['SKU ID', 'Ordered Quantity', 'Gross Merchandise Value']]

def get_recommendations(sku_id, data, top_n=20):
    data.fillna(0, inplace=True)
    scaler = MinMaxScaler()
    numerical_features = ['gross_merchandise_value','gift_wrap_expense', 'packaging_expense', 'handling_expense', 'shipping_expense']

    for col in numerical_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=numerical_features, inplace=True)
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    categorical_features = ['source', 'rto_status', 'cancellation_status', 'billing_address_state']
    data = pd.get_dummies(data, columns=categorical_features)

    product_profiles = data.groupby('sku_id')[numerical_features].mean().reset_index()
    feature_matrix = product_profiles.drop('sku_id', axis=1)
    similarity_matrix = cosine_similarity(feature_matrix)

    sku_id_to_index = {sku_id: index for index, sku_id in enumerate(product_profiles['sku_id'])}

    if sku_id not in sku_id_to_index:
        return pd.DataFrame(columns=['SKU ID', 'Ordered Quantity', 'Gross Merchandise Value'])

    index = sku_id_to_index[sku_id]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_sku_indices = [i for i, score in similarity_scores[1:top_n + 1]]
    recommended_skus = product_profiles.iloc[top_sku_indices]['sku_id'].tolist()

    ordered_quantity = data.groupby('sku_id')['ordered_quantity'].sum().reindex(recommended_skus, fill_value=0).tolist()
    gmv = data.groupby('sku_id')['gross_merchandise_value'].sum().reindex(recommended_skus, fill_value=0).tolist()

    recommendation_df = pd.DataFrame({
        'SKU ID': recommended_skus,
        'Ordered Quantity': ordered_quantity,
        'Gross Merchandise Value': gmv,
    })
    return recommendation_df[['SKU ID', 'Ordered Quantity', 'Gross Merchandise Value']]

def analyze_orders(data):
    data['order_date_time_utc'] = pd.to_datetime(data['order_date_time_utc'], errors='coerce')
    filtered_data = data[(data['order_date_time_utc'].dt.month.isin([2, 3])) & (data['order_date_time_utc'].dt.year == 2024)]
    filtered_data = filtered_data[filtered_data['sku_id'] != 0]

    sku_summary = filtered_data.groupby('sku_id').agg({
        'ordered_quantity': 'sum',
        'gross_merchandise_value': 'sum',
        'cancellation_status': lambda x: (x == 'TRUE').sum()
    }).reset_index()

    most_ordered = sku_summary.sort_values(by=['ordered_quantity'], ascending=False).head(10)
    least_ordered = sku_summary.sort_values(by=['ordered_quantity'], ascending=True).head(10)

    most_ordered_display = most_ordered[['sku_id', 'ordered_quantity', 'gross_merchandise_value']]
    least_ordered_display = least_ordered[['sku_id', 'ordered_quantity', 'gross_merchandise_value']]

    return most_ordered_display, least_ordered_display



@app.route('/recommend', methods=['POST'])
def recommend():
    area = request.form['area']
    sku_id = request.form['sku_id']
    method = request.form['method']

    # Select the recommendation method based on the user's choice
    if method == 'area_popularity':
        recommendations = recommend_items_with_similarity(area, sku_id, data)
        explanations = generate_area_based_explanations(sku_id, recommendations, data, area) if recommendations is not None else {}
    else:
        recommendations = get_recommendations(sku_id, data)
        explanations = generate_feature_based_explanations(sku_id, recommendations, data) if recommendations is not None else {}

    # Analyze the order data to get most and least ordered products
    most_ordered, least_ordered = analyze_orders(data)

    return jsonify({
        'recommendations': recommendations.to_dict(orient='records') if recommendations is not None else [],
        'most_ordered': most_ordered.to_dict(orient='records'),
        'least_ordered': least_ordered.to_dict(orient='records'),
        'explanations': explanations  # Make sure this is correctly passed
    })


def generate_feature_based_explanations(reference_sku, recommendations_df, data):
    """
    Generate explanations for each recommended product based on their similarity to the reference product.
    """
    # Select the features you want to include in explanations
    numerical_features = [
        'gross_merchandise_value', 'ordered_quantity', 'net_sales_before_tax',
        'gift_wrap_expense', 'packaging_expense', 'handling_expense', 'shipping_expense'
    ]

    # Filter the data to get profiles for the reference and recommended products
    reference_profile = data[data['sku_id'] == reference_sku][numerical_features].mean()
    explanations = {}

    for _, row in recommendations_df.iterrows():
        recommended_sku = row['SKU ID']
        recommended_profile = data[data['sku_id'] == recommended_sku][numerical_features].mean()
        
        # Initialize the explanation for the current recommended SKU
        explanation = []
        
        # Common features with similar values (difference < 0.1)
        common_features = [feature for feature in reference_profile.index if abs(reference_profile[feature] - recommended_profile[feature]) < 0.1]
        if common_features:
            explanation.append(f"The items are similar in the following features: {', '.join(common_features)}.")
            for feature in common_features:
                explanation.append(f"- {feature}: Both items have similar values ({reference_profile[feature]:.2f} vs {recommended_profile[feature]:.2f}).")
        
        # Features with notable differences (difference >= 0.1)
        different_features = [feature for feature in reference_profile.index if abs(reference_profile[feature] - recommended_profile[feature]) >= 0.1]
        for feature in different_features:
            difference = reference_profile[feature] - recommended_profile[feature]
            explanation.append(f"- {feature}: The reference item has a value of {reference_profile[feature]:.2f}, while the recommended item has {recommended_profile[feature]:.2f}.")
            
            if feature == 'gross_merchandise_value':
                explanation.append(f"The gross merchandise value differs by {abs(difference):.2f}, indicating a potential difference in revenue generation.")
            elif feature == 'ordered_quantity':
                explanation.append(f"The ordered quantity differs by {abs(difference):.2f}, indicating a different level of demand.")
            elif feature == 'shipping_expense':
                explanation.append(f"The shipping expenses differ by {abs(difference):.2f}, suggesting a difference in shipping costs.")
            elif feature == 'packaging_expense':
                explanation.append(f"Packaging expenses differ by {abs(difference):.2f}, reflecting different packaging requirements.")

        # Store explanation for each recommended SKU
        explanations[recommended_sku] = explanation

    return explanations

def generate_area_based_explanations(reference_sku, recommendations_df, data, area):
    """
    Generate explanations for area-based recommendations.
    """
    explanations = {}

    # Retrieve the reference item's profile for comparison
    reference_row = data[data['sku_id'] == reference_sku]
    
    # Check if the reference_row is empty
    if reference_row.empty:
        return explanations  # Return empty explanations if reference SKU is not found

    # Only consider numeric columns for calculating the mean
    reference_profile = reference_row.select_dtypes(include='number').mean()

    for _, row in recommendations_df.iterrows():
        recommended_sku = row['SKU ID']
        recommended_row = data[data['sku_id'] == recommended_sku]
        
        if recommended_row.empty:
            continue  # Skip if the recommended SKU is not found

        recommended_profile = recommended_row.select_dtypes(include='number').mean()
        
        explanation = [
            f"SKU {recommended_sku} is recommended because it is popular in {area}.",
            f"It has been ordered {row['Ordered Quantity']} times in this area."
        ]
        
        if row['Gross Merchandise Value'] != reference_profile['gross_merchandise_value']:
            explanation.append(
                f"The gross merchandise value for SKU {recommended_sku} is {row['Gross Merchandise Value']:.2f}, "
                f"which differs from the reference item's GMV of {reference_profile['gross_merchandise_value']:.2f}."
            )
        
        if row['Ordered Quantity'] != reference_profile['ordered_quantity']:
            explanation.append(
                f"The ordered quantity for SKU {recommended_sku} in {area} is {row['Ordered Quantity']}, "
                f"indicating a {('higher' if row['Ordered Quantity'] > reference_profile['ordered_quantity'] else 'lower')} demand compared to the reference item."
            )
        
        # Add explanations for shipping expense and cancellation status if available
        shipping_expense = recommended_row['shipping_expense'].sum()
        cancellation_status = recommended_row['cancellation_status'].mean()

        explanation.append(
            f"The shipping expense for SKU {recommended_sku} is {shipping_expense:.2f}."
        )
        explanation.append(
            f"The cancellation rate for SKU {recommended_sku} is {cancellation_status:.2f}."
        )
        
        explanations[recommended_sku] = explanation

    return explanations



# Configure upload folder and graph folder
# Configure upload folder and graph folder
UPLOAD_FOLDER = 'uploads'
GRAPHS_FOLDER = 'graphs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAPHS_FOLDER'] = GRAPHS_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load the dataset
    data = pd.read_csv(filepath)

    # Call functions to generate graphs
    graph_urls = generate_graphs(data)

    return jsonify(graph_urls)
 
def generate_graphs(data):
    graph_urls = []

    # Risk Management Graph
    risk_management_graph = plot_risk_management(data)
    graph_urls.append(risk_management_graph)

    # Portfolio Optimization Graph
    portfolio_optimization_graph = plot_portfolio_optimization(data)
    graph_urls.append(portfolio_optimization_graph)

    # Regulatory Compliance Graph
    regulatory_compliance_graph = plot_regulatory_compliance(data)
    graph_urls.append(regulatory_compliance_graph)

    # Customer Insights Graph
    customer_insights_graph = plot_customer_insights(data)
    graph_urls.append(customer_insights_graph)

    # Operational Efficiency Graph
    operational_efficiency_graph = plot_operational_efficiency(data)
    graph_urls.append(operational_efficiency_graph)

    return graph_urls



GRAPHS_FOLDER = './graphs'  # Update with your actual folder path

def plot_risk_management(data):
    # Ensure 'CPI Date' is in datetime format
    data['CPI Date'] = pd.to_datetime(data['CPI Date'], errors='coerce')
    
    # Drop rows with missing 'CPI Date' or 'CPI Index'
    data = data.dropna(subset=['CPI Date', 'CPI Index'])
    
    # Sort data by 'CPI Date'
    data = data.sort_values(by='CPI Date')
    
    # Plot CPI Index over time (Risk Management)
    plt.figure(figsize=(10, 6))
    plt.plot(data['CPI Date'], data['CPI Index'], marker='o', linestyle='-', color='b', label='CPI Index')
    
    # Set the title and labels
    plt.title('Risk Management: CPI Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('CPI Index')
    
    # Add a grid and legend
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'graph_1.png'))
    plt.close()
    
    return f'/graphs/graph_1.png'


GRAPHS_FOLDER = './graphs'  # Update this to your actual folder path

def plot_portfolio_optimization(data):
    # Ensure 'CPI Date' is in datetime format (optional, depending on analysis)
    data['CPI Date'] = pd.to_datetime(data['CPI Date'], errors='coerce')
    
    # Drop rows with missing 'Commodity' or 'CPI Index'
    data = data.dropna(subset=['Commodity', 'CPI Index'])
    
    # Group data by 'Commodity' and calculate the mean CPI Index for each commodity
    commodity_performance = data.groupby('Commodity')['CPI Index'].mean()
    
    # Plot commodity performance (CPI Index)
    plt.figure(figsize=(10, 6))
    commodity_performance.sort_values(ascending=False).plot(kind='bar', color='green')
    
    # Set the title and labels
    plt.title('Portfolio Optimization: Average CPI Index by Commodity')
    plt.xlabel('Commodity')
    plt.ylabel('Average CPI Index')
    
    # Add a grid
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'graph_2.png'))
    plt.close()
    
    return f'/graphs/graph_2.png'

def plot_regulatory_compliance(data):
    # Ensure the 'CPI Date' column is in datetime format
    data['CPI Date'] = pd.to_datetime(data['CPI Date'], errors='coerce')
    
    # Sorting data by date to ensure the graph is plotted chronologically
    data = data.sort_values(by='CPI Date')

    # Create a figure for Regulatory Compliance
    plt.figure(figsize=(10, 6))
    
    # Plotting the 'CPI Index Previous Year' against 'CPI Date'
    plt.plot(data['CPI Date'], data['CPI Index Previous Year'], color='red', label='CPI Index Previous Year')
    
    # Adding title and labels
    plt.title('Regulatory Compliance: CPI Index Previous Year Over Time')
    plt.xlabel('Date')
    plt.ylabel('CPI Index Previous Year')
    
    # Adding a grid for better readability
    plt.grid(True)
    
    # Adding a legend
    plt.legend(loc='best')
    
    # Save the figure as 'graph_3.png'
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'graph_3.png'))
    
    # Close the plot to avoid overlap
    plt.close()

    # Return the path to the saved graph
    return f'/graphs/graph_3.png'

def plot_customer_insights(data):
    plt.figure()
    data['Population Group'].value_counts().plot(kind='pie', autopct='%1.1f%%')  # Change as needed
    plt.title('Customer Insights')
    plt.ylabel('')
    plt.grid()
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'graph_4.png'))
    plt.close()
    return f'/graphs/graph_4.png'

def plot_operational_efficiency(data):
    plt.figure()
    data['CPI Index'].rolling(window=12).mean().plot()  # Example: update with your actual plotting logic
    plt.title('Operational Efficiency')
    plt.xlabel('Date')
    plt.ylabel('Rolling Mean CPI Index')
    plt.grid()
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'graph_5.png'))
    plt.close()
    return f'/graphs/graph_5.png'

@app.route('/graphs/<path:filename>')
def send_graph(filename):
    return send_from_directory(app.config['GRAPHS_FOLDER'], filename)

##Upload Data

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploadp', methods=['POST'])
def upload_filep():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

@app.route('/upload-url', methods=['POST'])
def upload_from_url():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        response = requests.get(url)
        response.raise_for_status()
        file_content = response.text

        # Determine file type based on content or URL extension
        filename = url.split('/')[-1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'w') as f:
            f.write(file_content)

        return jsonify({"message": "File from URL saved successfully", "filename": filename}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 400

# Route for fraud detection
@app.route('/fraud-detection', methods=['POST'])
def fraud_detection():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    dfc = df.copy()

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(
        df.select_dtypes(include=['float64', 'int64'])
    )

    # Identify categorical columns and apply label encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Standardize numerical features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Prepare features for model fitting
    feature_cols = df.columns.difference(['predicted_fraud'], sort=False)  # Exclude the predicted_fraud column
    model = IsolationForest(n_estimators=400, contamination=0.06, random_state=42, max_samples='auto')

    # Fit the model and make predictions using only the feature columns
    model.fit(df[feature_cols])

    # Convert DataFrame to NumPy array for decision_function
    df['anomaly_score'] = model.decision_function(df[feature_cols])

    # Get predictions
    df['predicted_fraud'] = model.predict(df[feature_cols])

    # Correctly map predictions to 1 for fraud (-1 from the model) and 0 for non-fraud (1 from the model)
    df['predicted_fraud'] = df['predicted_fraud'].map({-1: 1, 1: 0})

    # Normalize the decision function values to represent probabilities (0 to 1)
    df['non_fraud_probability'] = (df['anomaly_score'] - df['anomaly_score'].min()) / (
        df['anomaly_score'].max() - df['anomaly_score'].min())


    dfc['predicted_fraud'] = df['predicted_fraud']
    dfc['fraud_probability'] = 1-df['non_fraud_probability']

    # Get the count of fraud cases detected
    fraud_count = dfc['predicted_fraud'].sum()

    # If 'FraudFound_P' is present, calculate evaluation metrics
    if 'FraudFound_P' in dfc.columns:
        actual = dfc['FraudFound_P'].values
        predicted = dfc['predicted_fraud'].values
        try:
            accuracy = accuracy_score(actual, predicted)
            precision = precision_score(actual, predicted)
            recall = recall_score(actual, predicted)
            f1 = f1_score(actual, predicted)
            roc_auc = roc_auc_score(actual, predicted)
        except ValueError as e:
            return jsonify({'error': f'Error calculating metrics: {e}'}), 500
    else:
        accuracy = precision = recall = f1 = roc_auc = None

    # Save the result to a CSV file
    result_filename = 'fraud_detection_result.csv'
    dfc.to_csv(result_filename, index=False)

    return jsonify({
        'fraud_count': int(fraud_count),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'preview': dfc.head(30).to_dict(orient='records'),
        'result_file': result_filename
    })
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    dfc = df.copy()

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(
        df.select_dtypes(include=['float64', 'int64'])
    )

    # Identify categorical columns and apply label encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Standardize numerical features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Isolation Forest for anomaly detection
    model = IsolationForest(n_estimators=400, contamination=0.06, random_state=42, max_samples='auto')

    # Fit the model and make predictions
    model.fit(df)

    # Convert DataFrame to NumPy array for decision_function
    df_values = df.values
    df['anomaly_score'] = model.decision_function(df_values)

    df['predicted_fraud'] = model.fit_predict(df_values)

    # Map predictions to 1 for fraud (-1 from the model) and 0 for non-fraud (1 from the model)
    df['predicted_fraud'] = df['predicted_fraud'].map({1: 0, -1: 1})

    # Normalize the decision function values to represent probabilities (0 to 1)
    df['fraud_probability'] = (df['anomaly_score'] - df['anomaly_score'].min()) / (
            df['anomaly_score'].max() - df['anomaly_score'].min())

    dfc['predicted_fraud'] = df['predicted_fraud']
    dfc['fraud_probability'] = df['fraud_probability']

    # Get the count of fraud cases detected
    fraud_count = dfc['predicted_fraud'].sum()

    # If 'FraudFound_P' is present, calculate evaluation metrics
    if 'FraudFound_P' in dfc.columns:
        actual = dfc['FraudFound_P'].values
        predicted = dfc['predicted_fraud'].values
        try:
            accuracy = accuracy_score(actual, predicted)
            precision = precision_score(actual, predicted)
            recall = recall_score(actual, predicted)
            f1 = f1_score(actual, predicted)
            roc_auc = roc_auc_score(actual, predicted)
        except ValueError as e:
            return jsonify({'error': f'Error calculating metrics: {e}'}), 500
    else:
        accuracy = precision = recall = f1 = roc_auc = None

    # Save the result to a CSV file
    result_filename = 'fraud_detection_result.csv'
    dfc.to_csv(result_filename, index=False)

    return jsonify({
        'fraud_count': int(fraud_count),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'preview': dfc.head(10).to_dict(orient='records'),
        'result_file': result_filename
    })


# Route for downloading the result file
@app.route('/download', methods=['GET'])
def download_file():
    file_path = 'fraud_detection_result.csv'

    if os.path.exists(file_path):
        @after_this_request
        def remove_file(response):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
            return response

        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404



@app.route("/aivisualization")
def Ch():
    return aivisualization.index()

@app.route("/", methods=['GET', 'POST'])
def graph():
    return GraphGenerator.index()


@app.route('/recommendation')
def index():
    return render_template('recommendation.html')


# @app.route('/upload-url', methods=['POST'])
# def uploaddata():
#         return upload_data.upload_from_url()



# if __name__ == '__main__':
#     app.run(debug=True)
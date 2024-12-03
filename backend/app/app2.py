from flask import Flask, render_template, jsonify, request, send_file
import sqlite3
import os
import csv
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

app = Flask(__name__)

DATABASE = 'db/my_database.db'
CSV_PATH = './data.csv'  # Path to your CSV file

# Function to extract data from CSV into a DataFrame
def extract_csv_as_df(pathname: str) -> pd.DataFrame:
    """Extracts the content of the CSV into a pandas DataFrame."""
    return pd.read_csv(pathname)

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

# Helper function to generate graph based on user input
def generate_graph(df: pd.DataFrame, x_col: str, y_col: str, graph_type: str):
    plt.figure(figsize=(10, 6))
    if graph_type == 'bar':
        df.groupby(x_col)[y_col].sum().plot(kind='bar')
    elif graph_type == 'scatter':
        plt.scatter(df[x_col], df[y_col])
    elif graph_type == 'line':
        df.plot(x=x_col, y=y_col, kind='line')
    else:
        return None

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

# Chatbot route with graph generation
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_question = request.json.get('question')
    modified_question = f"{user_question} If this is related to graph generation, analyze which type of graph is best and provide the column details."

    # Read CSV data
    df = extract_csv_as_df(CSV_PATH)

    # Use Gemini AI to decide the graph type (for simplicity, we're using a basic rule-based logic here)
    if "graph" in user_question.lower():
        response = model.send_message(modified_question)

        # Assume Gemini AI suggests columns to be used for graphing (or we parse it ourselves)
        suggested_columns = extract_columns_from_question(user_question)  # Custom function to extract columns
        x_col, y_col = suggested_columns[0], suggested_columns[1]

        # Suggest graph type based on data type
        if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            graph_type = 'scatter'
        else:
            graph_type = 'bar'

        # Generate the graph
        graph_image = generate_graph(df, x_col, y_col, graph_type)
        return jsonify({'answer': f"Graph generated as {graph_type}.", 'graph': graph_image})
    
    # Fallback response if it's not related to graph generation
    response = model.send_message(modified_question)
    answer = response.text
    return jsonify({'answer': answer.strip()})

# Helper function to extract columns from the user's question (you can improve this as needed)
def extract_columns_from_question(question: str):
    # Dummy function - replace with better column extraction logic
    return ['column1', 'column2']

if __name__ == '__main__':
    app.run(debug=True)

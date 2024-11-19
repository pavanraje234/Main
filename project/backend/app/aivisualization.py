from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import csv
import google.generativeai as genai

app = Flask(__name__)

DATABASE = 'db/my_database.db'

# Function to extract data from CSV
def extract_csv(pathname: str) -> list[dict]:
    """Extracts the content of the CSV into a list of dictionaries with headers as keys."""
    data = []
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)  # Reads CSV into a list of dicts, with headers as keys
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

# Main dashboard route with dynamic cards
@app.route('/')
def index():
    # Fetch dynamic data for cards from the database
    # retailers=20
    # customers=20
    # escalations=20
    # issues=20
    retailers = query_db("SELECT COUNT(DISTINCT order_id) FROM Orders_New_query_2024_07_01", one=True)[0]
    customers = query_db("SELECT COUNT(DISTINCT order_id) FROM Orders_New_query_2024_07_01", one=True)[0]
    escalations = query_db("SELECT COUNT(*) FROM COGS_New_query_2024_07_01", one=True)[0]
    issues = query_db("SELECT COUNT(*) FROM COGS_New_query_2024_07_01", one=True)[0]

    # Pass the dynamic data into the template
    cards = {
        'retailers': retailers,
        'customers': customers,
        'escalations': escalations,
        'issues': issues
    }

    return render_template('index.html', cards=cards)

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
    csv_data = extract_csv("Orders_New_query_2024_07_01.csv")

    # Start a chat session with the model
    chat_session = model.start_chat(history=[{
        "role": "user",
        
        "parts": [{"text": str(csv_data)}]
    }])

    # Send the user's question to the model
    response = chat_session.send_message(modified_question)
    
    # Extract the response
    answer = response.text.split("*Output:")[1].split("")[0] if "Output:*" in response.text else response.text

    return jsonify({'answer': answer.strip()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
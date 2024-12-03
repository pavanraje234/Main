from flask import Flask, render_template, jsonify
import sqlite3

app = Flask(__name__)

DATABASE = 'db/my_database.db'

def query_db(query, args=(), one=False):
    con = sqlite3.connect(DATABASE)
    cur = con.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    con.close()
    return (rv[0] if rv else None) if one else rv

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True, port=5005)

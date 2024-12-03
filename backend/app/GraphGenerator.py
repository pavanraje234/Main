from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import re

app = Flask(__name__)

# Directory to save uploaded files and generated graphs
UPLOAD_FOLDER = 'uploads'
GRAPH_FOLDER = 'static/graphs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAPH_FOLDER'] = GRAPH_FOLDER

# Ensure the upload and graph folders exist
for folder in [UPLOAD_FOLDER, GRAPH_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to extract data from CSV into a DataFrame
def extract_csv_as_df(pathname: str) -> pd.DataFrame:
    """Extracts the content of the CSV into a pandas DataFrame."""
    return pd.read_csv(pathname)

def parse_user_input(query: str):
    """Parses the user's input for the X-axis, Y-axis, and graph type."""
    query = query.strip().lower()
    # Use regex to extract the graph type and columns
    match = re.match(r'(\w+)\s+vs\s+(\w+)\s+(bar|line|pie|scatter)', query.lower())
    if match:
        x_axis, y_axis, chart_type = match.groups()
        return chart_type, x_axis, y_axis
    else:
        return None, None, None

def validate_columns(x_axis: str, y_axis: str, df: pd.DataFrame):
    """Validates the X-axis and Y-axis column names against the DataFrame."""
    valid_columns = df.columns.tolist()
    error = None

    if x_axis not in valid_columns:
        error = f"Error: '{x_axis}' is not a valid column. Valid columns are: {', '.join(valid_columns)}."
    elif y_axis not in valid_columns:
        error = f"Error: '{y_axis}' is not a valid column. Valid columns are: {', '.join(valid_columns)}."
    
    return error

def generate_graph(df: pd.DataFrame, chart_type: str, x_axis: str, y_axis: str) -> str:
    """Generates a graph based on the user's query and saves it to the GRAPH_FOLDER."""
    try:
        timestamp = int(time.time())
        graph_filename = f"graph_{timestamp}.png"
        graph_path = os.path.join(app.config['GRAPH_FOLDER'], graph_filename)

        plt.figure(figsize=(10, 6))

        if chart_type == 'bar':
            plt.bar(df[x_axis], df[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f"{y_axis} by {x_axis}")

        elif chart_type == 'line':
            plt.plot(df[x_axis], df[y_axis], marker='o')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f"{y_axis} over {x_axis}")

        elif chart_type == 'pie':
            plt.pie(df[y_axis], labels=df[x_axis], autopct='%1.1f%%')
            plt.title(f"Pie chart of {y_axis}")

        elif chart_type == 'scatter':
            plt.scatter(df[x_axis], df[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(f"Scatter Plot of {y_axis} vs {x_axis}")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        return graph_filename

    except Exception as e:
        return f"Error generating graph: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Error: No file part in the request", 400

        file = request.files['file']
        if file.filename == '':
            return "Error: No selected file", 400

        if file and file.filename.endswith('.csv'):
            try:
                # Save the file and read CSV
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                df = extract_csv_as_df(filepath)

                # Retrieve the user's query
                query = request.form['query']

                # Parse user input
                chart_type, x_axis, y_axis = parse_user_input(query)

                if not chart_type or not x_axis or not y_axis:
                    return render_template('GraphGenerator.html', error_message="Invalid input. Please enter in the format: 'First_column vs Second_column bar/line/pie/scatter'.")

                # Validate the columns
                error = validate_columns(x_axis, y_axis, df)
                if error:
                    return render_template('GraphGenerator.html', error_message=error)

                # Generate the graph based on the parsed info
                graph_filename = generate_graph(df, chart_type, x_axis, y_axis)

                # Check if graph_filename is an error message
                if "Error" in graph_filename:
                    return render_template('GraphGenerator.html', error_message=graph_filename)

                # Show the graph to the user
                graph_url = url_for('static', filename=f'graphs/{graph_filename}')
                return render_template('GraphGenerator.html', graph_url=graph_url)

            except Exception as e:
                return f"Error: {str(e)}", 500

    return render_template('GraphGenerator.html')


if __name__ == '__main__':
    app.run(debug=True)
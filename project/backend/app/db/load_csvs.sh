#!/bin/bash

# Define the SQLite3 database name
DB_NAME="my_database.db"

# Function to load a CSV file into the SQLite3 database
load_csv_to_sqlite() {
    local csv_file="$1"
    local table_name=$(basename "$csv_file" .csv)

    # Remove the table if it already exists
    sqlite3 "$DB_NAME" "DROP TABLE IF EXISTS $table_name;"

    # Get the CSV header (first row)
    header=$(head -n 1 "$csv_file" | sed 's/,/ TEXT,/g') TEXT

    # Create the table with columns based on the header
    create_table_sql="CREATE TABLE $table_name ($header);"
    sqlite3 "$DB_NAME" "$create_table_sql"

    # Load the CSV data into the table, skipping the header
    sqlite3 "$DB_NAME" <<EOF
.mode csv
.import $csv_file $table_name
EOF

    echo "Loaded $csv_file into table $table_name"
}

# Loop over all CSV files in the current directory
for csv_file in *.csv; do
    [ -e "$csv_file" ] || continue
    load_csv_to_sqlite "$csv_file"
done

echo "All CSV files have been loaded into $DB_NAME"

# Check if the database was created and list tables
if [ -f "$DB_NAME" ]; then
    echo "Database $DB_NAME has been created."
    echo "Listing tables in $DB_NAME:"
    sqlite3 "$DB_NAME" ".tables"
else
    echo "Failed to create database $DB_NAME."
fi


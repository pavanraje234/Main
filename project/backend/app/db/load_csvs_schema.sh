#!/bin/bash

# Define the SQLite3 database name
DB_NAME="my_database.db"

# Function to load a CSV file into the SQLite3 database
load_csv_to_sqlite() {
    local csv_file="$1"
    local table_name=$(basename "$csv_file" .csv)

    # Remove the table if it already exists
    sqlite3 "$DB_NAME" "DROP TABLE IF EXISTS $table_name;"

    # Read the first line (header) to get column names
    header=$(head -n 1 "$csv_file")

    # Generate the CREATE TABLE statement with proper data types
    # Assuming all columns are TEXT by default, adjust as necessary
    create_table_sql="CREATE TABLE $table_name ("
    IFS=',' read -ra columns <<< "$header"
    for col in "${columns[@]}"; do
        if [[ "$col" =~ [0-9] ]]; then
            create_table_sql+="$col INTEGER,"
        else
            create_table_sql+="$col TEXT,"
        fi
    done
    create_table_sql=${create_table_sql%,}  # Remove trailing comma
    create_table_sql+=");"

    # Create the table with the generated SQL
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

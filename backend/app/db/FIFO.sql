-- Create the table to store the calculated COGS
DROP TABLE IF EXISTS calculated_cogs_2024_07_01;

CREATE TABLE calculated_cogs_2024_07_01 (
    order_id TEXT,
    item_name TEXT,
    order_date TEXT,
    ordered_quantity INTEGER,
    unit_price REAL,
    total_cogs REAL
);

-- -- Insert sample data into the COGS table (for demonstration purposes)
-- INSERT INTO COGS_New_query_2024_07_01 (item_name, product_category, cost_price, channel_name, created_at, Quantity) VALUES
-- ('ABCD', 'Category1', '100', 'Online', '2024-01-01', 10),
-- ('ABCD', 'Category1', '80', 'Online', '2024-06-01', 20);

-- -- Insert sample data into the Orders table (for demonstration purposes)
-- INSERT INTO Orders_New_query_2024_07_01 (order_line_item_id, source, order_id, ordered_quantity, tax_percent, net_sales_before_tax, gross_merchandise_value, sku_id, first_ordered_at, order_date_time_utc, refund_status, rto_status, cancellation_status, order_status, payment_status, billing_address_state, gift_wrap_expense, packaging_expense, handling_expense, shipping_expense, marketplace_expense, payment_gateway_expense, other_adjustments) VALUES
-- ('1', 'Online', 'ORD001', 5, '10', '500', '550', 'ABCD', '2024-01-02', '2024-01-02 10:00:00', 'None', 'None', 'None', 'Completed', 'Paid', 'CA', '0', '0', '0', '10', '0', '0', '0'),
-- ('2', 'Online', 'ORD002', 6, '10', '600', '660', 'ABCD', '2024-01-03', '2024-01-03 10:00:00', 'None', 'None', 'None', 'Completed', 'Paid', 'CA', '0', '0', '0', '10', '0', '0', '0'),
-- ('3', 'Online', 'ORD003', 10, '10', '1000', '1100', 'ABCD', '2024-01-05', '2024-01-05 10:00:00', 'None', 'None', 'None', 'Completed', 'Paid', 'CA', '0', '0', '0', '10', '0', '0', '0'),
-- ('4', 'Online', 'ORD004', 8, '10', '800', '880', 'ABCD', '2024-06-02', '2024-06-02 10:00:00', 'None', 'None', 'None', 'Completed', 'Paid', 'CA', '0', '0', '0', '10', '0', '0', '0');

-- Calculate COGS for each order item in a FIFO manner
WITH cte AS (
    SELECT o.order_id, c.item_name AS item_name, o.order_date_time_utc AS order_date, o.ordered_quantity AS order_units, 
           c.Quantity AS cogs_units, CAST(c.cost_price AS REAL) AS price, 
           SUM(o.ordered_quantity) OVER (PARTITION BY o.sku_id ORDER BY o.order_date_time_utc) AS cumulative_order_units,
           SUM(c.Quantity) OVER (PARTITION BY c.item_name ORDER BY c.created_at) AS cumulative_cogs_units,
           c.rowid AS cogs_rowid
    FROM Orders_New_query_2024_07_01 o
    JOIN COGS_New_query_2024_07_01 c ON o.sku_id = c.channel_name || '_' || c.sku_id
    ORDER BY o.order_date_time_utc, c.created_at
)
INSERT INTO calculated_cogs_2024_07_01 (order_id, item_name, order_date, ordered_quantity, unit_price, total_cogs)
SELECT order_id, item_name, order_date, order_units,
       price,
       order_units * price AS total_cogs
FROM cte
ORDER BY order_id, cogs_rowid;

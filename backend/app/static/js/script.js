$(document).ready(function() {
    // Fetch data and create Sales by Channel chart
    $.getJSON('/api/sales_by_channel', function(data) {
        var ctx = document.getElementById('salesByChannelChart').getContext('2d');
        var chartData = {
            labels: data.map(d => d.channel),
            datasets: [{
                label: 'Sales by Channel',
                data: data.map(d => d.sales),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        };
        new Chart(ctx, {
            type: 'bar',
            data: chartData
        });
    });

    // Fetch data and create Sales by SKU chart
    $.getJSON('/api/sales_by_sku', function(data) {
        var ctx = document.getElementById('salesBySkuChart').getContext('2d');
        var chartData = {
            labels: data.map(d => d.sku),
            datasets: [{
                label: 'Sales by SKU',
                data: data.map(d => d.sales),
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        };
        new Chart(ctx, {
            type: 'bar',
            data: chartData
        });
    });

    // Fetch data and populate SKU Profitability table
    $.getJSON('/api/sku_profitability', function(data) {
        var table = $('#skuProfitabilityTable').DataTable();
        table.clear();
        data.forEach(function(row) {
            table.row.add([row.sku, row.profitability]);
        });
        table.draw();
    });

    // Fetch data and create Profitability by Channel chart
    $.getJSON('/api/profitability_by_channel', function(data) {
        var ctx = document.getElementById('profitabilityByChannelChart').getContext('2d');
        var chartData = {
            labels: data.map(d => d.channel),
            datasets: [{
                label: 'Profitability by Channel',
                data: data.map(d => d.profitability),
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 1
            }]
        };
        new Chart(ctx, {
            type: 'bar',
            data: chartData
        });
    });
});

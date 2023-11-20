document.addEventListener('DOMContentLoaded', function () {
    var canvas = document.getElementById('valenceCanvas').getContext('2d');
    

    const graph = fetch('/get-data-graph-valence')
        .then(response => response.json())
        .then(data => {
            const labels = Object.keys(data);
            const values = Object.values(data);

            for (var i = 0; i < labels.length; i++) {
                labels[i] = parseFloat(labels[i]).toFixed(2).toString();
            }


            
            var myChart = new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '# of songs',
                        data: values,
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        );
    
    
});
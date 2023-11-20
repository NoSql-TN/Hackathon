document.addEventListener('DOMContentLoaded', function () {
    var canvas = document.getElementById('tempoCanvas').getContext('2d');
    

    const graph = fetch('/get-data-graph-tempo')
        .then(response => response.json())
        .then(data => {
            const labels = Object.keys(data);
            const values = Object.values(data);

            for (var i = 0; i < labels.length; i++) {
                labels[i] = parseFloat(labels[i]).toFixed(0).toString();
            }

            var temp;
            for (var i = 0; i < labels.length; i++) {
                for (var j = 0; j < labels.length; j++) {
                    if (parseFloat(labels[i]) < parseFloat(labels[j])) {
                        temp = labels[i];
                        labels[i] = labels[j];
                        labels[j] = temp;

                        temp = values[i];
                        values[i] = values[j];
                        values[j] = temp;
                    }
                }
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
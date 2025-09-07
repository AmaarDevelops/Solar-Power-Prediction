const form = document.getElementById('prediction-form')
const resultDiv = document.getElementById('result')
const resultText = document.getElementById('result-text')

form.addEventListener('submit', async(event) => {
    event.preventDefault();

    resultText.textContent = 'Predicting...'
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value,key) => {
        data[key] = parseFloat(value);
    });

    try{
        const response = await fetch('/predict' , {
            method : 'POST',
            headers : {
                'Content-Type' : 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`HTTP Error! status: ${response.status}`)
        }

        const result = await response.json()

        if (result.prediction !== undefined) {
            resultText.textContent = `${result.prediction.toFixed(2)} kW`;
        } else {
            resultText.textContent = 'Error: Prediction data not found';
            console.error('API response was missing prediction data :', result);
        }

    } catch(error) {
        resultText.textContent = 'Failed to get a prediction. Please try again.';
        console.error('Fetch Error :', error)
    }

});
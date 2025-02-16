let data = {
  "Username": "example_user",
  "Followers": 1000,
  "Posts": 50,
  "Sentiment": "Positive",
  "Reported": "No"
};

fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
  console.log(result); // Check if the response has the expected 'Prediction' field
  document.getElementById("predictionResult").innerText = result.Prediction; // Display result on the frontend
})
.catch(error => console.error('Error:', error));

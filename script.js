document.getElementById("prediction-form").addEventListener("submit", function (event) {
  event.preventDefault();

  // Simulate prediction logic
  let followers = document.getElementById("followers").value;
  let posts = document.getElementById("posts").value;
  let sentiment = document.getElementById("sentiment").value;
  let reported = document.getElementById("reported").value;

  let prediction = "legit"; // Default prediction

  // Example prediction rules
  if (reported === "Yes" || sentiment === "Negative" || followers < posts) {
      prediction = "deceptive";
  }

  // Redirect to result page
  if (prediction === "legit") {
      window.location.href = "legit.html";
  } else {
      window.location.href = "deceptive.html";
  }
});

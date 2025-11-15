document.getElementById("churnForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const jsonData = Object.fromEntries(formData.entries());

    // Convert numeric fields
    jsonData.SeniorCitizen = Number(jsonData.SeniorCitizen);
    jsonData.tenure = Number(jsonData.tenure);
    jsonData.MonthlyCharges = Number(jsonData.MonthlyCharges);
    jsonData.TotalCharges = Number(jsonData.TotalCharges);

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jsonData)
    });

    const result = await response.json();

    document.getElementById("result").innerHTML = `
        <strong>Model Used:</strong> ${result.model_used}<br>
        <strong>Prediction:</strong> ${result.prediction == 1 ? "Churn" : "Not Churn"}<br>
        <strong>Probability:</strong> ${(result.churn_probability * 100).toFixed(2)}%
    `;
});

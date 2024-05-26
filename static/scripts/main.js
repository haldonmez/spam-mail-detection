document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("spamForm");
    const resultDiv = document.getElementById("result");

    form.addEventListener("submit", async function(event) {
        event.preventDefault();
        
        const body = document.getElementById("emailBody").value;

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ body })
        });

        const result = await response.json();
        
        resultDiv.classList.remove("hidden");
        if (result.is_spam) {
            resultDiv.textContent = "The email is detected as spam.";
            resultDiv.style.backgroundColor = "#ffdddd";
            resultDiv.style.color = "#d9534f";
        } else {
            resultDiv.textContent = "The email is not spam.";
            resultDiv.style.backgroundColor = "#dff0d8";
            resultDiv.style.color = "#3c763d";
        }
    });
});

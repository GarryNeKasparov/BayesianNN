const button = document.getElementById("show-predictions");
const predictionsContainer = document.getElementById("predictions-container");

button.addEventListener("click", () => {
    if (predictionsContainer.style.display === "none" || predictionsContainer.style.display === "") {
        predictionsContainer.style.display = "flex";
        button.textContent = "Hide Individual Predictions";
    } else {
        predictionsContainer.style.display = "none";
        button.textContent = "Show Individual Predictions";
    }
});

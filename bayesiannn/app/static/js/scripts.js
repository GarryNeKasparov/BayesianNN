document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const response = await fetch("/predict", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);
        document.getElementById("output-image").src = imageUrl;
    } else {
        alert("Error: Unable to process the image.");
    }
});

const form = document.getElementById("upload-form");
const spinner = document.getElementById("loading-spinner");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const imageName = document.getElementById("image-name");

form.addEventListener("submit", () => {
    spinner.style.display = "flex";
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        fileLabel.style.backgroundColor = "#749436";
        fileLabel.style.color = "#fff";

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
        imageName.textContent = file.name;
    }
});

let videoStream;
let currentMode = 'none';
let uploadedFile = null;

const videoElement = document.getElementById('videoElement');
const imagePreview = document.getElementById('imagePreview');
const placeholder = document.getElementById('placeholder-text');
const canvas = document.getElementById('canvas');
const cameraBtn = document.getElementById('cameraBtn');
const previewContainer = document.getElementById('previewContainer');

/* ---------------- IMAGE UPLOAD ---------------- */
document.getElementById('imageInput').addEventListener('change', function (e) {
    if (e.target.files && e.target.files[0]) {
        uploadedFile = e.target.files[0];
        stopCamera();

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
            videoElement.hidden = true;
            placeholder.hidden = true;
            currentMode = 'image';
        };
        reader.readAsDataURL(uploadedFile);
    }
});

/* ---------------- CAMERA ---------------- */
function toggleCamera() {
    if (currentMode === 'camera') {
        capturePhoto();
    } else {
        startCamera();
    }
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then(stream => {
            videoStream = stream;
            videoElement.srcObject = stream;

            videoElement.hidden = false;
            imagePreview.hidden = true;
            placeholder.hidden = true;

            cameraBtn.innerHTML = '<i class="fas fa-camera"></i> Capture';
            cameraBtn.style.background = "#ffdd00";
            cameraBtn.style.color = "#000";

            currentMode = 'camera';
        })
        .catch(() => alert("Camera permission denied"));
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    videoElement.hidden = true;
    cameraBtn.innerHTML = '<i class="fas fa-camera"></i> Camera';
    cameraBtn.style.background = "";
    cameraBtn.style.color = "";
}

function capturePhoto() {
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);

    canvas.toBlob(blob => {
        uploadedFile = blob;
        imagePreview.src = URL.createObjectURL(blob);
        imagePreview.hidden = false;
        videoElement.hidden = true;
        stopCamera();
        currentMode = 'image';
    });
}

/* ---------------- CLASSIFICATION ---------------- */
async function classifyWaste() {
    if (!uploadedFile) {
        alert("Upload or capture an image first!");
        return;
    }

    const resultCard = document.getElementById('resultCard');
    const resultText = document.getElementById('wasteType');
    const progressFill = document.getElementById('progressFill');
    const confValue = document.getElementById('confValue');
    const hint = document.getElementById('disposalHint');

    // Reset UI
    resultCard.hidden = true;
    resultText.innerText = "Analyzing...";
    confValue.innerText = "0%";
    progressFill.style.width = "0%";

    // Start scanning animation
    previewContainer.classList.add('scanning');

    const formData = new FormData();
    formData.append("image", uploadedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Stop scanning
        previewContainer.classList.remove('scanning');
        resultCard.hidden = false;

        const confidence = Number(data.confidence).toFixed(1);
        progressFill.style.width = confidence + "%";
        confValue.innerText = confidence + "%";

        if (data.result === "Dry Waste") {
            resultText.innerText = "Dry Waste ♻️";
            resultText.style.color = "#80ffdb";
            progressFill.style.background = "#80ffdb";
            hint.innerText = "Disposal: Blue Bin (Paper, Plastic, Metal)";
        } else {
            resultText.innerText = "Wet Waste 🍌";
            resultText.style.color = "#ffdd00";
            progressFill.style.background = "#ffdd00";
            hint.innerText = "Disposal: Green Bin (Organic, Food Waste)";
        }

    } catch (error) {
        previewContainer.classList.remove('scanning');
        alert("Prediction failed. Check Flask server.");
        console.error(error);
    }
}

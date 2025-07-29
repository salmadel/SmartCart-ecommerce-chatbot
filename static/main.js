const sendBtn = document.getElementById("send-btn");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const imageUpload = document.getElementById("image-upload");
const recordBtn = document.getElementById("record-btn");

const previewArea = document.querySelector(".preview-area");
const previewMessage = document.querySelector(".preview-message");
const previewImage = document.getElementById("preview-image");

let mediaRecorder;
let audioChunks = [];
let selectedImage = null;
let isRecording = false;
let currentStream = null;

function addMessage(content, sender, isImage = false) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);

    if (isImage) {
        const textDiv = document.createElement("div");
        textDiv.innerText = content.text || "";
        textDiv.style.marginBottom = "5px";
        msg.appendChild(textDiv);

        const img = document.createElement("img");
        img.src = content.image;
        img.style.maxWidth = "200px";
        img.style.borderRadius = "8px";
        msg.appendChild(img);
    } else {
        msg.innerText = content;
    }

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.onclick = async function () {
    const text = userInput.value.trim();

    if (!text && !selectedImage) return;

    if (selectedImage) {
        addMessage({ text: text, image: selectedImage }, "user", true);
    } else if (text) {
        addMessage(text, "user");
    }

    const payload = {
        query: text || null,
        image_base64: selectedImage ? selectedImage.split(",")[1] : null,
    };

    userInput.value = "";
    selectedImage = null;
    imageUpload.value = "";

    previewArea.style.display = "none";
    previewMessage.innerText = "";
    previewImage.src = "";

    const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const data = await response.json();
    addMessage(data.answer || "Something went wrong!", "bot");
};

imageUpload.onchange = function () {
    const file = imageUpload.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = function () {
        selectedImage = reader.result;

        previewImage.src = selectedImage;
        previewMessage.innerText = userInput.value.trim() || "";
        previewArea.style.display = "flex";
    };
    reader.readAsDataURL(file);
};

userInput.addEventListener("input", () => {
    if (selectedImage) {
        previewMessage.innerText = userInput.value.trim() || "";
    }
});

recordBtn.onclick = async function () {
    if (isRecording) {
        mediaRecorder.stop();
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        isRecording = false;
        recordBtn.classList.remove("recording");
        console.log("Recording stopped");
    } else {
        if (!navigator.mediaDevices) {
            alert("Your browser does not support audio recording.");
            return;
        }
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(currentStream, { mimeType: "audio/webm" });

            mediaRecorder.start();
            isRecording = true;
            recordBtn.classList.add("recording");
            console.log("Recording started");

            audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", async () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const reader = new FileReader();

                reader.onloadend = async function () {
                    const base64Audio = reader.result.split(",")[1];

                    const response = await fetch("/api/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ audio_base64: base64Audio }),
                    });

                    const data = await response.json();
                    console.log("Response from server:", data);

                    const audioURL = URL.createObjectURL(audioBlob);

                    const msg = document.createElement("div");
                    msg.classList.add("message", "user");

                    const audioElem = document.createElement("audio");
                    audioElem.controls = true;
                    audioElem.src = audioURL;

                    msg.appendChild(audioElem);

                    chatBox.appendChild(msg);
                    chatBox.scrollTop = chatBox.scrollHeight;

                    addMessage(data.answer || "Something went wrong!", "bot");
                };

                reader.readAsDataURL(audioBlob);
                audioChunks = [];
            });



        } catch (error) {
            console.error("Error accessing microphone:", error);
            alert("Could not access microphone.");
        }
    }
};

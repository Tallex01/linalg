const btn = document.getElementById("load-btn");
const result = document.getElementById("result");
const labelSpan = document.getElementById("label");
const predictionSpan = document.getElementById("prediction");
const smileySpan = document.getElementById("smiley");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Loading...";

    try {
        const res = await fetch("/image");
        const data = await res.json();

        console.log("Received data:", data);
        console.log("Label:", data.label, "Type:", typeof data.label);
        console.log("Guessed:", data.guessed_digit, "Type:", typeof data.guessed_digit);

        labelSpan.textContent = data.label;
        predictionSpan.textContent = data.guessed_digit;

        // Display smiley based on correctness
        const isCorrect = parseInt(data.label) === parseInt(data.guessed_digit);
        console.log("Is correct:", isCorrect);

        if (isCorrect) {
            smileySpan.textContent = ":)";
            smileySpan.className = "smiley correct";
        } else {
            smileySpan.textContent = ":(";
            smileySpan.className = "smiley incorrect";
        }

        // Draw the 28x28 grayscale image onto the canvas
        const imageData = ctx.createImageData(28, 28);
        for (let row = 0; row < 28; row++) {
            for (let col = 0; col < 28; col++) {
                const value = Math.round(data.image[row][col] * 255);
                const i = (row * 28 + col) * 4;
                imageData.data[i] = value;
                imageData.data[i + 1] = value;
                imageData.data[i + 2] = value;
                imageData.data[i + 3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);

        result.hidden = false;
    } catch (error) {
        console.error("Error loading image:", error);
        alert("Error loading image: " + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Load Random Digit";
    }
});

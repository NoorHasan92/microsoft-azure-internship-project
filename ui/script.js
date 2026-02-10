// ui/script.js

let typingTimeout = null; // Store timeout ID to stop it if needed

function typeWriter(text, element, speed = 15) {
    let i = 0;
    element.innerHTML = ""; // Clear existing text
    
    // Clear any previous typing animation
    if (typingTimeout) clearTimeout(typingTimeout);

    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            typingTimeout = setTimeout(type, speed);
        }
    }
    type();
}

async function analyzeText() {
    const textArea = document.querySelector("textarea");
    const button = document.querySelector("button");
    const resultBox = document.getElementById("resultBox");
    const aiBox = document.getElementById("aiResponseContainer");
    const aiTextTarget = document.getElementById("aiTextBody");
    const text = textArea.value.trim();

    if (!text) {
        resultBox.style.display = "block";
        resultBox.className = "result"; // Reset color
        resultBox.innerHTML = "⚠️ Please enter text.";
        return;
    }

    // Reset UI state
    resultBox.style.display = "block";
    resultBox.className = "result loading";
    resultBox.innerHTML = "Processing...";
    
    // Hide AI box smoothly
    aiBox.style.display = "none";
    aiTextTarget.innerHTML = "";
    
    button.disabled = true;
    button.innerText = "Analyzing...";

    try {
        const response = await fetch("https://mental-health-dl-api.onrender.com/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        // 1. Show Technical Result
        // Clean up the class name to match CSS (remove 'Critical' or 'Medium')
        let colorClass = "low";
        if (data.priority === "Critical" || data.priority === "High") colorClass = "high";
        else if (data.priority === "Medium") colorClass = "moderate";

        resultBox.className = "result " + colorClass;
        
        // Format the score nicely
        resultBox.innerHTML = `
            <div><strong>Risk:</strong> ${data.risk_label}</div>
            <div><strong>Score:</strong> ${data.risk_score}%</div>
            <div><strong>Priority:</strong> ${data.priority}</div>
        `;

        // 2. Reveal AI Guidance with Animation
        aiBox.style.display = "block";
        typeWriter(data.explanation, aiTextTarget, 15);

    } catch (error) {
        console.error(error);
        resultBox.className = "result high";
        resultBox.innerHTML = "❌ Server Error. Is the backend running?";
    } finally {
        button.disabled = false;
        button.innerText = "Analyze";
    }
}
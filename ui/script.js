// ui/script.js

let typingTimeout = null;

// ---------------------------------------------
// Typewriter Effect
// ---------------------------------------------
function typeWriter(text, element, speed = 15) {
    let i = 0;
    element.innerHTML = "";

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

// ---------------------------------------------
// Main Analysis Function
// ---------------------------------------------
async function analyzeText() {
    const textArea = document.querySelector("textarea");
    const button = document.querySelector("button");
    const resultBox = document.getElementById("resultBox");
    const aiBox = document.getElementById("aiResponseContainer");
    const aiTextTarget = document.getElementById("aiTextBody");
    const text = textArea.value.trim();

    // Reset animation + state
    requestAnimationFrame(() => {
        resultBox.classList.add("show");
    });


    aiBox.style.display = "none";
    aiBox.classList.remove("show");
    aiTextTarget.innerHTML = "";

    if (!text) {
        resultBox.className = "result";
        resultBox.innerHTML = "⚠️ Please enter text.";

        requestAnimationFrame(() => {
            resultBox.classList.add("show");
        });

        return;
    }

    // Show processing state
    resultBox.className = "result loading";
    resultBox.innerHTML = "Processing...";

    requestAnimationFrame(() => {
        resultBox.classList.add("show");
    });

    button.disabled = true;
    button.innerText = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error("Server response not OK");
        }

        const data = await response.json();
        // Show detailed section
        const detailSection = document.getElementById("detailedAnalysis");
        detailSection.style.display = "block";

        // Populate symptoms
        const symptomList = document.getElementById("symptomList");
        symptomList.innerHTML = "";

        data.detected_symptoms.forEach(symptom => {
            const pill = document.createElement("div");
            pill.className = "symptom-pill";
            pill.innerText = `${symptom.label} (${Math.round(symptom.confidence * 100)}%)`;
            symptomList.appendChild(pill);
        });

        // Pattern
        document.getElementById("patternBox").innerHTML =
            `<strong>Pattern:</strong> ${data.psychological_pattern}`;

        // Severity
        const severityBox = document.getElementById("severityBox");
        severityBox.innerText = data.symptom_severity;

        severityBox.className = "severity-badge";

        if (data.symptom_severity.includes("Severe")) {
            severityBox.classList.add("severity-severe");
        } else if (data.symptom_severity.includes("Moderate")) {
            severityBox.classList.add("severity-moderate");
        } else {
            severityBox.classList.add("severity-low");
        }

        // Emergency Support
        const emergencyBox = document.getElementById("emergencyBox");

        if (data.emergency_support) {
            emergencyBox.style.display = "block";
            emergencyBox.innerHTML = `
        <strong>Immediate Support Available</strong><br><br>
        🇮🇳 India: ${data.emergency_support.india_helpline}<br>
        🇺🇸 US: ${data.emergency_support.us_helpline}<br><br>
        ${data.emergency_support.message}
        <a href="https://nhm.hp.gov.in/storage/app/media/uploaded-files/Mental%20Health%20Support%20Numbers.pdf"
           target="_blank"
           class="pdf-link">
           📄 View Mental Health Support Numbers (India)
        </a>
    `;
        } else {
            emergencyBox.style.display = "none";
        }

        // Disclaimer
        document.getElementById("disclaimerBox").innerText = data.disclaimer;


        // Determine color class
        let colorClass = "low";
        if (data.priority === "Critical" || data.priority === "High") {
            colorClass = "high";
        } else if (data.priority === "Medium") {
            colorClass = "moderate";
        }

        // Update result content
        resultBox.className = "result " + colorClass;
        resultBox.innerHTML = `
    <div><strong>Risk:</strong> ${data.risk_label}</div>
    <div><strong>Priority:</strong> ${data.priority}</div>
    <div><strong>System Confidence:</strong> ${data.system_confidence}</div>

    <div class="confidence-wrapper">
        <div class="confidence-label">
            Risk Score: ${data.risk_score}%
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: 0%"></div>
        </div>
    </div>
`;

        // Animate confidence bar
        setTimeout(() => {
            const fill = document.querySelector(".confidence-fill");
            if (fill) {
                fill.style.width = data.risk_score + "%";
            }
        }, 200);


        // Re-trigger animation
        resultBox.classList.remove("show");
        requestAnimationFrame(() => {
            resultBox.classList.add("show");
        });

        // Stagger AI guidance reveal (calm delay)
        setTimeout(() => {
            aiBox.style.display = "block";
            requestAnimationFrame(() => {
                aiBox.classList.add("show");
            });
            typeWriter(data.explanation, aiTextTarget, 15);
        }, 350);

    } catch (error) {
        console.error(error);

        resultBox.className = "result high";
        resultBox.innerHTML = "❌ Server Error. Please try again.";

        resultBox.classList.remove("show");
        requestAnimationFrame(() => {
            resultBox.classList.add("show");
        });
    } finally {
        button.disabled = false;
        button.innerText = "Analyze Emotional Distress";
    }
}

async function sendMessage() {

    const input = document.getElementById("user-input");

    const message = input.value;

    if (message.trim() === "") {
        return;
    }

    const chatBox = document.getElementById("chat-box");

    // USER MESSAGE

    chatBox.innerHTML += `
        <div class="message user">
            ${message}
        </div>
    `;

    input.value = "";

    chatBox.scrollTop = chatBox.scrollHeight;

    // TYPING EFFECT

    const typingId = Date.now();

    chatBox.innerHTML += `
        <div class="message bot" id="${typingId}">
            Thinking...
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;

    // API CALL

    const response = await fetch("/chat", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            message: message
        })
    });

    const data = await response.json();

    // REMOVE THINKING

    document.getElementById(typingId).remove();

    // BOT RESPONSE

    chatBox.innerHTML += `
        <div class="message bot">
            ${data.response}
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;
}

// ENTER KEY SUPPORT

document
.getElementById("user-input")
.addEventListener("keypress", function(event) {

    if (event.key === "Enter") {
        sendMessage();
    }
});
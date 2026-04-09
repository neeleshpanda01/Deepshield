// Background service worker
chrome.runtime.onInstalled.addListener(() => {
  console.log("DeepShield extension installed");
  chrome.storage.local.set({
    enabled: true,
    threshold: 0.85,
    backend_url: "http://localhost:5000"
  });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "predict") {
    // Forward prediction request to backend
    chrome.storage.local.get(["backend_url"], (result) => {
      const backend_url = result.backend_url || "http://localhost:5000";

      console.log(`[Background] Sending request to ${backend_url}`);
      fetch(`${backend_url}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: request.image })
      })
        .then(response => {
          console.log(`[Background] Response status: ${response.status}`);
          return response.json();
        })
        .then(data => {
          console.log("[Background] Data received", data);
          sendResponse(data);
        })
        .catch(error => {
          console.error("[Background] Fetch error:", error);
          sendResponse({ error: error.message });
        });
    });

    return true; // Keep channel open for async response
  }
});

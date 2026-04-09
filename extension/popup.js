// Popup script

const toggleBtn = document.getElementById('toggle');
const thresholdInput = document.getElementById('threshold');
const backendUrlInput = document.getElementById('backend-url');

// Load settings
chrome.storage.local.get(['enabled', 'threshold', 'backend_url'], (result) => {
  const enabled = result.enabled !== false;
  const threshold = result.threshold || 0.85;
  const backend_url = result.backend_url || 'http://localhost:5000';
  
  toggleBtn.textContent = enabled ? 'Disable Detection' : 'Enable Detection';
  toggleBtn.className = enabled ? 'toggle-btn toggle-on' : 'toggle-btn toggle-off';
  thresholdInput.value = threshold;
  backendUrlInput.value = backend_url;
});

// Toggle detection
toggleBtn.addEventListener('click', () => {
  chrome.storage.local.get(['enabled'], (result) => {
    const newState = !result.enabled;
    chrome.storage.local.set({ enabled: newState });
    
    toggleBtn.textContent = newState ? 'Disable Detection' : 'Enable Detection';
    toggleBtn.className = newState ? 'toggle-btn toggle-on' : 'toggle-btn toggle-off';
    
    // Notify content script
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach(tab => {
        chrome.tabs.sendMessage(tab.id, { type: 'toggle' }).catch(() => {});
      });
    });
  });
});

// Save threshold
thresholdInput.addEventListener('change', () => {
  chrome.storage.local.set({ threshold: parseFloat(thresholdInput.value) });
});

// Save backend URL
backendUrlInput.addEventListener('change', () => {
  chrome.storage.local.set({ backend_url: backendUrlInput.value });
});

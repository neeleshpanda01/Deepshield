// Content script - runs in Jitsi page context
console.error("!!! DEEPSHIELD CONTENT SCRIPT STARTED !!! - If you see this, the extension is loaded on this page.");


const ALERT_THRESHOLD = 0.85;
const FRAME_INTERVAL = 10; // Analyze every Nth frame
const SMOOTHING_WINDOW = 5;

let frameCount = 0;
let predictionHistory = [];
let enabled = true;
let lastAlertTime = 0;
const ALERT_COOLDOWN = 2000; // 2 seconds

// Create overlay container
const overlayContainer = document.createElement('div');
overlayContainer.id = 'deepshield-overlay';
overlayContainer.style.cssText = `
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 10000;
`;
document.body.appendChild(overlayContainer);

// Create alert banner
const alertBanner = document.createElement('div');
alertBanner.id = 'deepshield-alert';
alertBanner.style.cssText = `
  position: fixed;
  top: 20px;
  right: 20px;
  background: #ff4444;
  color: white;
  padding: 15px 20px;
  border-radius: 8px;
  display: none;
  z-index: 10001;
  font-family: Arial, sans-serif;
  font-weight: bold;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  max-width: 300px;
`;
document.body.appendChild(alertBanner);

// Create status indicator
const statusIndicator = document.createElement('div');
statusIndicator.id = 'deepshield-status';
statusIndicator.style.cssText = `
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #333;
  color: #fff;
  padding: 10px 15px;
  border-radius: 6px;
  font-family: Arial, sans-serif;
  font-size: 12px;
  z-index: 10001;
  text-align: center;
`;
statusIndicator.innerHTML = '🟢 DeepShield: Active';
document.body.appendChild(statusIndicator);

function updateStatus(label, color) {
  statusIndicator.innerHTML = `${color} DeepShield: ${label}`;
}

function showAlert(confidence) {
  const now = Date.now();
  if (now - lastAlertTime < ALERT_COOLDOWN) return;

  lastAlertTime = now;
  alertBanner.innerHTML = `⚠️ DEEPFAKE DETECTED<br>Confidence: ${(confidence * 100).toFixed(1)}%`;
  alertBanner.style.display = 'block';

  setTimeout(() => {
    alertBanner.style.display = 'none';
  }, 3000);
}

let currentAnalyzedVideo = null;

function getLargestVisibleVideo() {
  const videos = Array.from(document.querySelectorAll('video'));
  if (videos.length === 0 && frameCount % 50 === 0) {
    console.log("DeepShield: No <video> elements found yet.");
  }

  let maxArea = 0;
  let largestVideo = null;

  for (const video of videos) {
    // Check if visible
    const rect = video.getBoundingClientRect();
    const area = rect.width * rect.height;

    // Debug log for potential candidates
    if (area > 10000 && frameCount % 50 === 0) {
      console.log(`DeepShield: Candidate video found. ID: ${video.id}, Size: ${rect.width}x${rect.height}, Visible: ${video.style.display !== 'none'}`);
    }

    // Basic visibility check
    if (rect.width <= 10 || rect.height <= 10 || video.style.display === 'none' || video.style.visibility === 'hidden') {
      continue;
    }

    // Attempt to filter out self-view if possible (Jitsi often flips self-view or puts it in a smaller container)
    // But relying on "Largest" is the most robust heuristic for "Active Speaker" or "Featured Content"

    if (area > maxArea) {
      maxArea = area;
      largestVideo = video;
    }
  }

  if (!largestVideo && videos.length > 0 && frameCount % 50 === 0) {
    console.log("DeepShield: Videos found but none met criteria.");
  }

  return largestVideo;
}

function updateVisualHighlight(newVideo) {
  // Remove border from previous
  if (currentAnalyzedVideo && currentAnalyzedVideo !== newVideo) {
    // Only remove if it was our border (simple reset)
    currentAnalyzedVideo.style.outline = '';
  }

  if (newVideo) {
    // Add visual indicator (Blue outline = Analysis Active)
    if (newVideo !== currentAnalyzedVideo) {
      newVideo.style.outline = '3px solid #3498db';
      currentAnalyzedVideo = newVideo;
    }
  } else {
    currentAnalyzedVideo = null;
  }
}

function captureVideoFrame() {
  try {
    const video = getLargestVisibleVideo();
    updateVisualHighlight(video);

    if (!video) return null;

    const canvas = document.createElement('canvas');
    // Ensure we have valid dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]; // Return base64 without prefix
  } catch (e) {
    console.error('Failed to capture frame:', e);
    return null;
  }
}
function processPrediction(result) {
  if (result.error) {
    console.error('Prediction error:', result.error);
    updateStatus('Error', '🔴');
    return;
  }

  // Add to history
  predictionHistory.push({
    prediction: result.prediction,
    confidence: result.prediction === 0 ? result.fake_confidence : result.real_confidence
  });

  // Keep only recent predictions
  if (predictionHistory.length > SMOOTHING_WINDOW) {
    predictionHistory.shift();
  }

  // Calculate average
  const avgPrediction = Math.round(
    predictionHistory.reduce((sum, p) => sum + p.prediction, 0) / predictionHistory.length
  );
  const avgConfidence = predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / predictionHistory.length;

  // Update status
  if (avgPrediction === 0) {
    updateStatus(`Fake: ${(avgConfidence * 100).toFixed(0)}%`, '🔴');
  } else {
    updateStatus(`Real: ${(avgConfidence * 100).toFixed(0)}%`, '🟢');
  }

  // Alert if high confidence fake
  if (avgPrediction === 0 && avgConfidence >= ALERT_THRESHOLD) {
    showAlert(avgConfidence);
  }
}

async function analyzeFrame() {
  if (!enabled) {
    if (frameCount % 50 === 0) console.log("DeepShield: Disabled");
    return;
  }

  frameCount++;
  if (frameCount % 10 === 0) console.log(`DeepShield: Frame ${frameCount}. Checking videos...`);
  if (frameCount % FRAME_INTERVAL !== 0) return;

  const imageB64 = captureVideoFrame();
  if (!imageB64) {
    if (frameCount % (FRAME_INTERVAL * 5) === 0) {
      console.log("DeepShield: No valid video frame captured.");
    }
    return;
  }

  console.log("DeepShield: Frame captured. Sending to background...");

  // Send to background script
  chrome.runtime.sendMessage(
    { type: "predict", image: imageB64 },
    (response) => {
      console.log("DeepShield: Received response from background:", response);
      if (chrome.runtime.lastError) {
        console.error("DeepShield: Runtime error:", chrome.runtime.lastError);
      }
      if (response) {
        processPrediction(response);
      }
    }
  );
}

// Start analysis loop
setInterval(analyzeFrame, 100); // Check every 100ms

// Listen for messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "toggle") {
    enabled = !enabled;
    updateStatus(enabled ? 'Active' : 'Disabled', enabled ? '🟢' : '⚫');
    sendResponse({ enabled });
  }
});

console.log("DeepShield content script loaded");

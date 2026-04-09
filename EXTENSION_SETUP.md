# DeepShield Browser Extension Setup

## Installation & Usage

### Step 1: Install Flask and CORS
```powershell
pip install flask flask-cors
```

### Step 2: Start the Backend Server
```powershell
C:\Users\Asus\AppData\Local\Programs\Python\Python311\python.exe backend_server.py
```

You should see:
```
Model loaded (ResNet50)
Using device: cpu
Server ready on http://localhost:5000
```

### Step 3: Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked**
4. Select the `extension` folder in your DeepShield project
5. Extension should appear in your toolbar

### Step 4: Use with Jitsi

1. Go to any Jitsi meeting: https://meet.jitsi.org
2. Allow camera/microphone permissions
3. The DeepShield indicator will appear in bottom right
4. When someone is detected as deepfake:
   - ⚠️ Red alert banner appears (top right)
   - Status shows "Fake: XX%"
   - Console logs the detection

### Step 5: Configure Settings

Click the extension icon to open popup:
- **Toggle Detection**: Enable/Disable analysis
- **Alert Threshold**: Confidence level for alerts (default 0.85 = 85%)
- **Backend URL**: API endpoint (default http://localhost:5000)

## File Structure

```
extension/
├── manifest.json       # Extension config
├── background.js       # Service worker
├── content.js         # Video capture & analysis
├── popup.html         # Settings UI
├── popup.js           # Settings logic
└── styles.css         # Styling
```

## How It Works

1. **Content Script** (`content.js`):
   - Runs in Jitsi page
   - Captures video frames every 10 frames
   - Sends to backend for prediction

2. **Background Worker** (`background.js`):
   - Forwards predictions to backend API
   - Manages settings & state

3. **Backend** (`backend_server.py`):
   - Flask REST API
   - Loads ResNet50 model
   - Analyzes frames and returns predictions

4. **Popup** (`popup.html`/`popup.js`):
   - Settings management
   - Enable/disable detection
   - Configure threshold & backend URL

## Troubleshooting

**No alerts appearing?**
- Check if backend is running: `http://localhost:5000/health`
- Check browser console for errors (F12)
- Ensure Jitsi video element is loading

**Backend not connecting?**
- Verify Flask server is running
- Check `backend_url` in extension settings
- Ensure port 5000 is not blocked

**Extension not loading?**
- Go to `chrome://extensions/` and check for errors
- Reload extension (refresh button)
- Clear browser cache

## Notes

- Extension works best in Chrome (Manifest v3)
- Firefox support requires minor changes
- Backend must be running on same machine for localhost
- For remote backend, update backend_url in settings

## Testing

To test without Jitsi:
```powershell
# Use the Python test script
C:\Users\Asus\AppData\Local\Programs\Python\Python311\python.exe realtime_analysis.py
```

# How to Test the BMI Prediction App Locally

This guide explains how to run the **Expo Go frontend** and **Flask backend** to test BMI prediction using an image of a face.

---

## 1. Run the Frontend (React Native with Expo)

### Prerequisites

- Node.js installed
- `expo-cli` installed globally or use via `npx`
- An iOS or Android smartphone

### Steps

1. Navigate to your frontend project directory:

   ```bash
   cd bmi-app-frontend
   ```

2. Start the Expo development server:

   ```bash
   npx expo start
   ```

3. Install the **Expo Go** app on your mobile device:
   - [iOS – App Store](https://apps.apple.com/app/expo-go/id982107779)
   - [Android – Google Play](https://play.google.com/store/apps/details?id=host.exp.exponent)

4. Scan the QR code displayed in your terminal or browser using the Expo Go app.

**Note:** Ensure both your computer and mobile device are connected to the same Wi-Fi network.

---

## 2. Run the Backend (Flask + PyTorch Model)

### Prerequisites

- Python 3.7 or later
- Required Python libraries:
  - `flask`, `flask-cors`, `torch`, `torchvision`, `Pillow`

### Steps

1. Navigate to your backend folder:

   ```bash
   cd model_backend
   ```

2. Install Python dependencies:

   ```bash
   pip install flask flask-cors torch torchvision Pillow
   ```

3. Run the Flask server:

   ```bash
   python app.py
   ```

4. Your backend should be running at:

   ```
   http://127.0.0.1:5000
   ```

**If using a real phone for testing**, replace `127.0.0.1` in your React Native code with your computer's local IP address (e.g., `http://192.xx.xx.xx:5000`).

---

## Troubleshooting

### Port 5000 is Already in Use

If you are a mac user, see macOS guideline for freeing port 5000

To check what is using port 5000:

```bash
lsof -i :5000
```

To terminate the process:

```bash
kill -9 <PID>
```

Or change the port in `app.py`:

```python
app.run(debug=True, port=5001)
```

### macOS: Freeing Port 5000 (AirPlay Conflict)

If port 5000 is used by AirPlay Receiver:

1. Open **System Settings**
2. Go to **General > AirDrop & Handoff**
3. Disable **AirPlay Receiver**

This will release port 5000 for use by Flask.

### Mobile App Cannot Connect to Server

- Use your computer’s IP address instead of `127.0.0.1` in the frontend
- Confirm that both devices are on the same network
- Make sure your firewall allows connections to port 5000

---

## You're All Set

Now you can upload or capture a face image and receive a predicted BMI result from your trained model.

---
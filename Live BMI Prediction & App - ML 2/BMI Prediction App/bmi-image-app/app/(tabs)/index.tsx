import React, { useState } from 'react';
import { View, Button, Image, StyleSheet, Text, ActivityIndicator, Alert, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [image, setImage] = useState(null);
  const [bmiResult, setBmiResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = 'http://127.0.0.1/predict'; // Your Flask server IP

  const pickImageFromLibrary = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      sendImageToServer(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Camera permission is required!');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      sendImageToServer(result.assets[0].uri);
    }
  };

  const sendImageToServer = async (uri) => {
    setLoading(true);
    setBmiResult(null);
    try {
      const formData = new FormData();
      const filename = uri.split('/').pop();
      const match = /\.(\w+)$/.exec(filename || '');
      const fileType = match ? `image/${match[1]}` : `image`;

      formData.append('file', {
        uri,
        name: filename,
        type: fileType,
      });

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      if (response.ok) {
        setBmiResult(result.predicted_bmi);
      } else {
        Alert.alert('Error', result.error || 'Prediction failed');
      }
    } catch (err) {
      console.error(err);
      Alert.alert('Error', 'Failed to send image to server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>BMI Prediction</Text>
      <Text style={styles.subtitle}>Upload or capture a face image to estimate BMI.</Text>

      <View style={styles.buttonRow}>
        <TouchableOpacity onPress={pickImageFromLibrary} style={styles.button}>
          <Text style={styles.buttonText}>Choose from Album</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={takePhoto} style={styles.button}>
          <Text style={styles.buttonText}>Take a Photo</Text>
        </TouchableOpacity>
      </View>

      {image && (
        <View style={styles.resultCard}>
          <Image source={{ uri: image }} style={styles.image} />
          {loading ? (
            <ActivityIndicator size="large" style={{ marginTop: 16 }} />
          ) : bmiResult !== null ? (
            <View style={{ alignItems: 'center' }}>
              <Text style={styles.resultLabel}>Predicted BMI</Text>
              <Text style={styles.resultValue}>{bmiResult}</Text>
            </View>
          ) : null}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafe',
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    marginBottom: 24,
    textAlign: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#2c3e50',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    marginHorizontal: 5,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  resultCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    marginTop: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
  },
  image: {
    width: 240,
    height: 240,
    borderRadius: 16,
    marginBottom: 16,
  },
  resultLabel: {
    fontSize: 16,
    color: '#34495e',
    marginBottom: 6,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#27ae60',
  },
});
# VietSpeech AI — Mobile App Guide

## Setup

### Prerequisites

- Flutter SDK ≥ 3.0.0 ([flutter.dev](https://flutter.dev/docs/get-started/install))
- Android Studio or VS Code with Flutter extension
- Android device or emulator (API 21+)
- iOS device or simulator (iOS 14+, macOS only)

### Install & Run

```bash
cd mobile/flutter_app

# Install dependencies
flutter pub get

# Verify setup
flutter doctor

# Run on connected device
flutter run

# Run on specific device
flutter run -d emulator-5554  # Android emulator
flutter run -d "iPhone 15"   # iOS simulator
```

---

## Configuration

### Backend URL

Edit `lib/core/api/api_client.dart`:

```dart
static const String _defaultBaseUrl = 'http://10.0.2.2:8000'; // Android emulator
// static const String _defaultBaseUrl = 'http://localhost:8000'; // iOS simulator
// static const String _defaultBaseUrl = 'https://your-domain.com'; // Production
```

---

## Features

| Screen | Route | Description |
|--------|-------|-------------|
| Home | `/` | Feature overview, KPI stats |
| ASR | `/asr` | Record audio → transcribe |
| Dialect | `/dialect` | Record audio → identify province/region |
| TTS | `/tts` | Type text → synthesize + play audio |
| Translate | `/translate` | Vi↔Ede/Bahnar/En translation |
| Login | `/login` | JWT auth (optional) |

---

## Build for Production

```bash
# Android APK
flutter build apk --release

# Android App Bundle (Google Play)
flutter build appbundle --release

# iOS (macOS only)
flutter build ipa --release
```

---

## Permissions

**Android** (`android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
```

**iOS** (`ios/Runner/Info.plist`):
```xml
<key>NSMicrophoneUsageDescription</key>
<string>Cần quyền truy cập microphone để ghi âm giọng nói</string>
```

---

## Offline Mode (TFLite)

The app includes a `tflite_flutter` dependency for on-device inference:

1. Export ONNX model to TFLite: (notebook 08, add TFLite export step)
2. Place model in `mobile/flutter_app/assets/models/`
3. TFLite inference falls back automatically when backend is unreachable

**Note**: TFLite dialect model (~500MB) is not bundled by default. Enable in `lib/core/offline/`.

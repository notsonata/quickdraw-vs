# Testing

## Strategy

Automated tests cover pure, local behavior:

- Stub classifier JSON contract.
- TFLite label loading and missing-model fallback.
- TFLite inference smoke testing when downloaded model files are present.
- Preprocessing output shape, normalization, and profile-specific behavior.
- Preprocessing polarity for TFLite: white background, black strokes.
- Speech gate timing and limiting rules.
- Web canvas round timer state and main-loop auto-end behavior.
- Gemma vision result validation and Gemma speech priority.
- Preset response generation.

Manual testing is required for OS screen capture and local audio output because those depend on the desktop session and installed audio stack.

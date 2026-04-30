# Decisions

## 2026-04-30: Keep Google Meet Out of Process

Google Meet is only the social layer. The app does not automate Meet, inspect Meet participants, or detect winners.

## 2026-04-30: Stub Classifier Is Required

The target model `zarqankhn/quickdraw-345-tflite` is not installed. The app must run without a model by using a valid stub classifier so capture, gating, response, logging, and TTS behavior can be tested.

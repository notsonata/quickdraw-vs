# API Contracts

## Classifier Result

```json
{
  "top1": "cat",
  "confidence": 0.72,
  "top3": [
    ["cat", 0.72],
    ["dog", 0.18],
    ["rabbit", 0.06]
  ]
}
```

## Speech Gate Result

```json
{
  "round_active": true,
  "top1": "bottlecap",
  "confidence": 0.68,
  "top3": [
    ["bottlecap", 0.68],
    ["circle", 0.14],
    ["gear", 0.08]
  ],
  "spoken_label": "bottlecap",
  "alternate_label": null,
  "stable_ms": 620,
  "should_speak": true,
  "reason": "stable_confident_guess",
  "ai_guesses_this_round": 1
}
```

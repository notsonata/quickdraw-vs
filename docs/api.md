# API Contracts

## Classifier Result

```json
{
  "source": "quickdraw",
  "top1": "cat",
  "confidence": 0.72,
  "top3": [
    ["cat", 0.72],
    ["dog", 0.18],
    ["rabbit", 0.06]
  ]
}
```

`source` is optional for legacy classifier backends. Gemma vision results set `"source": "gemma"` and use the same `top1`, `confidence`, `top3`, and `top5` shape after validating the label against the QuickDraw label file.

## Speech Gate Result

```json
{
  "round_active": true,
  "source": "quickdraw",
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

## Web Canvas Round Status

`/status`, `/events`, and `/event` responses include:

```json
{
  "round": {
    "round_active": true,
    "remaining_sec": 42
  }
}
```

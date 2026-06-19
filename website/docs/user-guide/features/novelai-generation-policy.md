---
title: NovelAI Generation Policy
---

# NovelAI Generation Policy V1

This document defines the Hermes standard operating policy for NovelAI image generation. It is a policy document only; applying it to runtime configuration or profile prompts requires a separate approved change.

## Scope

Applies to NovelAI / NAI image generation defaults and shared image-quality baseline guidance across Hermes image workflows.

This document does not authorize:

- image generation
- NSFW generation
- high-resolution generation
- upscale
- config changes
- profile/SOUL changes

NovelAI live generation requires a separate explicit operator approval and runtime
call flag:

```text
LIVE_GENERATION_REQUIRES_APPROVAL
live_generation_approved=True
```

## NAI default policy

| Setting | Default |
| --- | --- |
| Add Quality Tags | `OFF` |
| Undesired Content Preset | `NONE` |
| Sampler | `DPM++ SDE` |
| SMEA | `ON` |
| DYN | `ON` |

## NAI default negative prompt

Use this negative prompt as the NovelAI default baseline:

```text
normal quality,
bad quality,
low quality,
worst quality,
lowres,
bad anatomy,
bad hands,
malformed hands,
malformed fingers,
missing fingers,
extra digits,
fewer digits,
watermark,
signature,
username,
text,
blurry,
duplicate,
mutation,
deformed,
disfigured,
extra arms,
extra legs,
bad feet,
bad proportions,
JPEG artifacts,
chromatic aberration,
scan artifacts
```

## NAI resolution policy

### Default allowed range

The default allowed range is `SAFE_1024_RANGE`:

- `1024x1024` or smaller
- equivalent pixel-count range at or below roughly `1,048,576` pixels

Examples:

- `1024x1024`
- `832x1216`
- `768x1344`

### High-resolution range

Anything above the `1024x1024` pixel budget is high-resolution operation.

Examples:

- `1536x1536`
- `2048x2048`
- upscale / high-resolution upscale

Policy:

```text
HIGH_RES_REQUIRES_APPROVAL
```

High-resolution operation may consume Anlas or other paid resources. Obtain explicit user approval before running it.

## Global image baseline

The following negative prompt terms are recommended as a shared baseline for ComfyUI, Angelica, Palette, and NovelAI workflows:

```text
bad anatomy,
bad hands,
malformed hands,
malformed fingers,
missing fingers,
extra digits,
fewer digits,
bad feet,
bad proportions
```

These terms target anatomy and extremity defects without over-constraining facial expression.

## Eye-related terms are not global defaults

Do not include these as global baseline negatives:

```text
bad eyes,
cross-eye,
lazy eye,
asymmetrical eyes
```

Reason: broad eye negatives can reduce character attractiveness, stylization, and expressive facial variation. Use eye-related negatives only when a specific model, checkpoint, prompt family, or failed output demonstrates a recurring eye defect.

## Apply readiness

This policy is apply-ready as a Hermes standard policy document.

Runtime application still requires a separate implementation step, for example:

- NovelAI adapter default parameters
- profile-specific prompt templates
- workflow-specific generation presets
- UI/config exposure

No runtime config or SOUL/profile file is modified by this document.

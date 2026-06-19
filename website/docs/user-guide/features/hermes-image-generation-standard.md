---
title: Hermes Image Generation Standard
---

# Hermes Image Generation Standard V1

This document extends the NovelAI Generation Policy V1 into the Hermes-wide image generation standard. It is a policy document only; applying it to runtime configuration, provider defaults, workflow presets, or profile/SOUL prompts requires a separate approved implementation change.

## Scope

Applies to Hermes image generation workflows across:

- NovelAI / NAI / Seir
- ComfyUI / Angelica
- GPT Image / Palette
- shared artifact publishing, sidecar, hash, Slack delivery, and NAS hook conventions

This document does not authorize image generation, NAI calls, NSFW generation, high-resolution generation, upscale, config changes, profile/SOUL changes, commit, or push.

NovelAI live generation remains approval-gated even when the provider is
installed:

```text
LIVE_GENERATION_REQUIRES_APPROVAL
live_generation_approved=True
```

## 1. Global Image Baseline

Hermes image workflows should use a shared anatomy/extremity quality baseline when the backend supports negative prompts or equivalent prompt constraints.

Recommended shared negative baseline:

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

This baseline may be referenced by NovelAI, ComfyUI, and Palette/GPT Image workflows where technically appropriate.

Do not include the following eye-related terms as global defaults:

```text
bad eyes,
cross-eye,
lazy eye,
asymmetrical eyes
```

Reason: broad eye negatives can reduce character attractiveness, stylization, and expressive facial variation. Eye-specific negatives should be applied only when a specific model, checkpoint, prompt family, or failed output shows a recurring eye defect.

## 2. NAI / Seir Policy

NAI-specific defaults apply only to NovelAI / NAI / Seir workflows.

### NAI default settings

| Setting | Default |
| --- | --- |
| Add Quality Tags | `OFF` |
| Undesired Content Preset | `NONE` |
| Sampler | `DPM++ SDE` |
| SMEA | `ON` |
| DYN | `ON` |

`DPM++ SDE`, `SMEA`, `DYN`, `Add Quality Tags OFF`, and `Undesired Content Preset NONE` are NAI-only settings. Do not apply them to ComfyUI, Angelica, GPT Image, or Palette unless those systems explicitly implement compatible semantics.

### NAI default negative prompt

Use this as the NovelAI / Seir default negative prompt baseline:

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

## 3. ComfyUI / Angelica Policy

ComfyUI / Angelica workflows may reference the Global Image Baseline for anatomy, hand, finger, foot, and proportion defect suppression.

Rules:

- Do not inherit NAI-only sampler/preset settings.
- Do not treat `DPM++ SDE`, `SMEA`, or `DYN` as Hermes-wide settings.
- Keep ComfyUI sampler, scheduler, checkpoint, LoRA, ControlNet, and workflow settings backend-specific.
- Preserve existing ComfyUI artifact conventions: published PNG outputs, sidecar metadata, manifest/integrity records, and NAS/Slack delivery readiness when delivery is requested.

## 4. GPT Image / Palette Policy

GPT Image / Palette workflows may reference the Global Image Baseline as quality guidance, but not all terms map to hard negative-prompt controls.

Rules:

- Do not inherit NAI-only sampler/preset settings.
- Express the anatomy/hand/foot baseline as prompt guidance or review criteria when the backend does not support explicit negative prompts.
- Keep GPT Image provider defaults and safety behavior provider-specific.
- Maintain the same artifact, sidecar, hash, Slack, and NAS delivery expectations for final user-facing outputs.

## 5. Resolution Policy

### NAI default operation

NAI / Seir default operation uses:

```text
SAFE_1024_RANGE
```

Allowed by default:

- `1024x1024` or smaller
- equivalent pixel-count range at or below roughly `1,048,576` pixels

Examples:

- `1024x1024`
- `832x1216`
- `768x1344`

### Hermes-wide high-resolution classification

For all Hermes image workflows, operations above the normal backend-safe baseline should be treated as high-resolution or resource-intensive until proven otherwise.

NAI-specific threshold:

```text
1024x1024 초과 = HIGH_RES_REQUIRES_APPROVAL
```

Examples:

- `1536x1536`
- `2048x2048`
- upscale / high-resolution upscale

## 6. Artifact / Sidecar / Hash Policy

Final generated image artifacts should use the Hermes production artifact pattern where possible:

```text
/Users/hermes/HermesWork/Image/<Backend>/<run_id>/
  image_000.png
  sidecar/
    request.json
    response.json
    metadata.json
    manifest.json
    integrity.json
```

Policy:

- Final user-facing images should be published under `HermesWork/Image` before Slack/NAS delivery.
- Keep source generation paths separate from published delivery paths.
- Record request/response metadata in sidecars.
- Record artifact SHA256 and integrity status.
- Verify PNG signature and hash before declaring an artifact delivery-ready.
- For NovelAI, raw response bodies such as `response.bin` or `response.zip` are debug-only and must not be preserved by default; record raw response metadata in `response.json` instead.

## 7. Slack Delivery Policy

Slack delivery should use published Hermes image artifacts only.

Rules:

- Use `media_files=[published_png]` for image attachment delivery.
- Do not deliver raw profile `generated/` paths directly.
- Published paths must pass the Slack media candidate validator.
- Raw/generated paths outside allowed publish roots should remain blocked.
- Delivery verification should distinguish text-message success from actual file upload success.
- For thread delivery, preserve thread target metadata when available.

PASS evidence should include, when delivery is performed:

- send result success
- file attachment present
- expected file count
- filename/kind
- target channel/thread evidence

## 8. NAS Hook Policy

NAS sync should operate on published Hermes image artifacts, not raw source-generation files.

Rules:

- Queue or run NAS hooks only for published `HermesWork/Image/...` artifacts.
- Verify NAS mirror paths directly, for example `/Volumes/Hermes/image/<Backend>/<run_id>/...`.
- Verify mirrored PNG SHA256 matches the published/local artifact.
- If sidecars are updated after an initial NAS sync, re-run the NAS hook before declaring NAS sidecar evidence current.

## 9. High Resolution Approval Policy

High-resolution or resource-intensive generation requires explicit user approval before execution.

For NAI / Seir:

```text
HIGH_RES_REQUIRES_APPROVAL
```

Applies to:

- any generation above `SAFE_1024_RANGE`
- `1536x1536`
- `2048x2048`
- upscale / high-resolution upscale
- any setting likely to consume Anlas or other paid resources

For ComfyUI / Angelica and GPT Image / Palette, apply the same approval principle when the operation materially increases cost, runtime, GPU load, queue time, or paid API usage.

## Apply readiness

This standard is ready as a Hermes-wide policy reference.

Runtime application remains a separate change and may require backend-specific implementation in:

- NovelAI adapter defaults
- ComfyUI workflow presets
- GPT Image / Palette prompt templates
- artifact publishing pipeline
- Slack delivery integration
- NAS hook integration
- profile-specific operating documents

No runtime config or profile/SOUL file is modified by this document.

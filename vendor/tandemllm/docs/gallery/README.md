# Gallery

The dashboard at 1440×900 and on a 390×844 phone, dark theme. `dashboard/e2e/readme-shots.spec.ts` renders the pictures from the dashboard's mock, and `dashboard/scripts/readme-shots.sh` converts them to WebP. The Setup Guide speeds come from the measured rows in `models/*/model.yaml`. All other numbers come from the mock.

## Chat

![The Chat tab: a web page the model wrote in its sandbox, open in the preview pane, with the context panel](chat-preview.webp)

The model writes `site/index.html` in its sandbox, the preview pane renders it, and the context panel shows what fills the window.

## Setup Guide

![The models step: one tile per model with its measured speed](setup-models.webp)

The models step: one tile per model, each with its measured speed in tok/s.

![Nemotron 3.5 Bolt's speed table: tok/s and KL for each size](setup-speed.webp)

Each size's tok/s and KL, here Nemotron 3.5 Bolt.

![Community weights: turboderp's Qwen3.8-Flash-Next EXL3 packs](setup-community.webp)

Community weights with their maker: turboderp's Flash-Next packs.

## Performance and Usage

![Performance: live decode speed and each running request](performance.webp)

Live decode speed and every running request.

![Performance history: decode speed, time to first token, prefill, tokens per block, draft acceptance](performance-history.webp)

History: decode speed, first-token time, prefill, tokens per block, draft acceptance.

![Usage: tokens per day over a year](usage.webp)

Tokens per day for a year.

## Phone

<p>
  <img src="phone-chat-answer.webp" width="300" alt="The Chat tab on a phone: an answer with code">
  &nbsp;&nbsp;
  <img src="phone-chat-run.webp" width="300" alt="The Chat tab on a phone: a run with a tool call">
</p>

The Chat tab on a phone: an answer with code, and a run calling a tool.

---
title: IMS Backend Worker
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

This Hugging Face Space runs a continuous background worker and scheduler for the IMS Semantic Search project.

It runs:
1. `scheduler.py` - Background scheduler with a dummy webserver on port 7860 to satisfy Space health checks.
2. `scraper.py` - Parses standard IMS notices and downloads PDFs, limiting to recent 30 to stay within limits.
3. `worker/worker.py` - Performs OCR on PDFs, chunks them, embeds them with all-MiniLM-L6-v2, and saves to Supabase pgvector.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

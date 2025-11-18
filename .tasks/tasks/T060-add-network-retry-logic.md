---
id: T060
title: Add Network Retry Logic for Gist/HF Downloads
status: pending
priority: 2
agent: backend
dependencies: [T051, T052, T053]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [reliability, network, phase2, refactor, enhancement]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - template.ipynb
  - training.ipynb

est_tokens: 6000
actual_tokens: null
---

## Description

Add exponential backoff retry logic to network requests (Gist downloads, HF model downloads, W&B logging) to handle transient failures. Colab's network occasionally drops, causing notebook failures despite code being correct.

Current state: Network requests fail immediately on timeout/connection error. Users must manually restart cells, wasting time.

Target state: `@retry_with_backoff` decorator retries failed requests up to 3 times with exponential backoff (1s, 2s, 4s delays). Works with requests, urllib, HF hub.

## Business Context

**Why This Matters:** Colab network flakiness causes ~5-10% of notebook runs to fail. Retry logic reduces failures to <1%, improving user experience.

**Priority:** P2 - Reliability improvement for production use.

## Acceptance Criteria

- [ ] `retry_with_backoff()` decorator created with max_retries=3, backoff_factor=2
- [ ] Applied to Gist download functions in template.ipynb
- [ ] Applied to HF model downloads in training.ipynb
- [ ] Applied to W&B logging (optional - W&B has internal retries)
- [ ] Logs retry attempts: "⚠️ Network error, retrying (1/3)..."
- [ ] Validation: Simulate network error (disconnect WiFi), verify retry succeeds after reconnect
- [ ] Unit test: Mock request failures, verify retry logic

## Technical Implementation

```python
import time
import functools

def retry_with_backoff(max_retries=3, backoff_factor=2, exceptions=(Exception,)):
    """Retry function with exponential backoff on specified exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    print(f"⚠️  {func.__name__} failed: {e}")
                    print(f"   Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, exceptions=(requests.RequestException, urllib.error.URLError))
def download_gist_file(gist_id, filename):
    """Download file from Gist with retry logic."""
    url = f"https://gist.githubusercontent.com/.../raw/{filename}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text
```

## Dependencies

**Hard Dependencies:** None
**Blocks:** None

## Completion Checklist

- [ ] Retry decorator implemented with exponential backoff
- [ ] Applied to Gist downloads
- [ ] Applied to HF downloads
- [ ] Logs retry attempts clearly
- [ ] Unit tests pass

**Definition of Done:** Network errors auto-retry, <1% failure rate from transient issues.

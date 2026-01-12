# Summarizer System Documentation

**Last Updated**: Sprint 6 (January 2026)  
**Status**: Production Ready

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Metrics](#metrics)
- [API Reference](#api-reference)
- [Performance & Monitoring](#performance--monitoring)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

The Summarizer system is a production-grade "LLM as a Judge" evaluation platform that extracts structured insights from conversation data through multi-metric extraction with native structured output support.

### Key Features

- **Multi-Metric Extraction**: Separate LLM calls for summary, outcome, tags, and classification
- **Provider Agnostic**: Supports Ollama, OpenAI, LiteLLM, and Open WebUI providers
- **Structured Outputs**: Native JSON mode support with automatic retry logic
- **Quality Validation**: Hallucination detection via keyword overlap analysis
- **Drop-off Detection**: Identifies abandoned conversations
- **Production Resilience**: Exponential backoff, graceful degradation, partial failure handling
- **Comprehensive Monitoring**: Real-time metrics tracking with detailed logging

---

## Architecture

### System Flow

1. User configures provider and model via Admin UI
2. Conversation data is processed through metric extractors
3. Each metric makes independent LLM call with JSON mode
4. Quality validation checks for hallucinations
5. Results stored in SQLite with JSON metadata
6. Monitoring tracks performance and failures

### Core Components

- **Metric Extractors** (`backend/metrics/`): Specialized prompts per metric
- **Provider Registry** (`backend/provider_registry.py`): Multi-provider support
- **Quality Validation** (`backend/metrics/validation.py`): Hallucination detection
- **Monitoring System** (`backend/monitoring.py`): Performance tracking
- **Storage Layer** (`backend/storage.py`): SQLite + JSON metadata

---

## Configuration

### Environment Variables

```bash
# Provider URLs
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...
LITELLM_API_KEY=your-key
LITELLM_API_BASE=http://localhost:4000

# Retry Configuration
SUMMARIZER_USE_EXPONENTIAL_BACKOFF=true
SUMMARIZER_RETRY_MAX_ATTEMPTS=5
SUMMARIZER_MAX_RETRIES=2

# Quality & Validation
SUMMARIZER_ENABLE_QUALITY_VALIDATION=true
SUMMARIZER_MIN_KEYWORD_OVERLAP=0.15
SUMMARIZER_ENABLE_DROPOFF_DETECTION=true

# Logging
SUMMARIZER_ENABLE_DETAILED_LOGGING=false
SUMMARIZER_LOG_RETENTION_HOURS=72
```

### Admin UI Setup

1. Navigate to **Admin → Summarizer**
2. Select provider (Ollama, OpenAI, LiteLLM, Open WebUI)
3. Choose model from dropdown
4. Set temperature (0.0-2.0, recommended: 0.2)
5. Select metrics to extract
6. Test connection
7. Save configuration

---

## Metrics

### Summary
One-line conversation summary with quality validation.

**Output**: `{"summary": "Python async error handling discussion"}`

### Outcome
Multi-factor scoring: completeness, accuracy, helpfulness (1-5 scale).

**Output**: 
```json
{
  "outcome_score": 4,
  "completeness": 4,
  "accuracy": 5,
  "helpfulness": 4,
  "reasoning": "Complete answer with examples"
}
```

### Tags
Topic classification tags (3-7 per conversation).

**Output**: `{"tags": ["python", "async", "error-handling"]}`

### Classification
Domain and resolution status categorization.

**Output**:
```json
{
  "domain": "technical",
  "interaction_type": "qa",
  "resolution_status": "resolved"
}
```

---

## API Reference

### Metric Extraction

**POST `/api/v1/metrics/extract`**
Extract metrics from conversation.

Request:
```json
{
  "chat_id": "abc123",
  "metrics": ["summary", "outcome"],
  "force": false
}
```

### Admin Configuration

**GET `/api/v1/admin/summarizer/settings`**
Get current configuration.

**POST `/api/v1/admin/summarizer/settings`**
Update configuration.

**POST `/api/v1/admin/summarizer/test-connection`**
Test provider connectivity.

### Monitoring

**GET `/api/v1/admin/summarizer/monitoring/overall`**
Overall statistics.

**GET `/api/v1/admin/summarizer/monitoring/by-metric`**
Per-metric breakdown.

**GET `/api/v1/admin/summarizer/monitoring/recent-failures`**
Recent extraction failures.

---

## Performance & Monitoring

### Performance Characteristics

- **Throughput**: 10-20 conversations/second (single metric)
- **Latency**: 200-500ms per metric
- **Memory**: ~0.5MB per conversation
- **Concurrency**: 2-3x speedup with asyncio.gather()

### Monitoring Dashboard

Access at **Admin → Summarizer → Monitoring**:

1. **Overall Stats**: Success rate, avg latency, total tokens
2. **Per-Metric Breakdown**: Success/failure counts, latency
3. **Recent Failures**: Error messages, retry counts
4. **Export Logs**: Download for offline analysis

### Detailed Logging

Enable with `SUMMARIZER_ENABLE_DETAILED_LOGGING=true`.
Logs saved to `logs/summarizer/{date}.jsonl`.

---

## Troubleshooting

### Provider Unavailable
- Verify environment variables
- Check service is running
- Restart backend

### Model Not Found
- Click "Refresh" in UI
- Verify model exists (`ollama list`)
- Check spelling (case-sensitive)

### JSON Parsing Failures
- Increase `SUMMARIZER_PARSE_RETRY_ATTEMPTS`
- Use larger models
- Enable detailed logging

### Low Quality Summaries
- Adjust temperature (0.2-0.5)
- Use larger models
- Check quality validation threshold

### High Failure Rate
- Check provider rate limits
- Increase retry attempts
- Verify network connectivity

### Slow Performance
- Use faster models (llama3.2:3b)
- Enable concurrent extraction
- Reduce temperature

---

## Best Practices

### Model Selection

- **Summary**: Small models (llama3.2:3b), temp 0.2-0.3
- **Outcome**: Medium models (llama3.2:7b), temp 0.1-0.2
- **Tags**: Small models, temp 0.0-0.1
- **Classification**: Small models, temp 0.0

### Configuration

**Development**:
```bash
SUMMARIZER_ENABLE_DETAILED_LOGGING=true
SUMMARIZER_MAX_RETRIES=1
```

**Production**:
```bash
SUMMARIZER_ENABLE_DETAILED_LOGGING=false
SUMMARIZER_MAX_RETRIES=2
SUMMARIZER_ENABLE_GRACEFUL_DEGRADATION=true
```

### Performance Optimization

1. Extract only needed metrics
2. Use concurrent processing (asyncio.gather)
3. Start with smallest capable model
4. Cache partial results with graceful degradation

### Quality Assurance

1. Monitor success rates (target: >95%)
2. Review sample results periodically
3. A/B test different models
4. Adjust quality thresholds as needed

### Security & Privacy

1. Disable detailed logging in production
2. Store API keys in environment variables
3. Implement data retention policies
4. Consider GDPR/CCPA compliance

---

## Appendix

### Sprint Timeline

- **Sprint 1**: Provider infrastructure, JSON mode, retry logic
- **Sprint 2**: Multi-metric extraction, selective execution
- **Sprint 3**: Quality validation, drop-off detection
- **Sprint 4**: Admin UI, metric configuration
- **Sprint 5**: Monitoring infrastructure, detailed logging
- **Sprint 6**: Integration tests, load tests, documentation

### Related Documentation

- [CLAUDE.md](../CLAUDE.md): Architecture and development patterns
- [README.md](../README.md): Quick start guide
- [.env.example](../.env.example): Environment variables

---

**Version**: Sprint 6  
**Status**: Production Ready  
**Last Updated**: January 2026

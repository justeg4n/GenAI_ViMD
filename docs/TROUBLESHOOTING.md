# VietSpeech AI — Troubleshooting Guide

## Common Issues

---

### Backend Won't Start

**Symptom**: `uvicorn` crashes immediately on startup.

**Check 1**: Database connection
```bash
# Verify PostgreSQL is running
docker ps | grep postgres

# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Fix: Ensure DATABASE_URL in .env matches your DB setup
```

**Check 2**: Redis connection
```bash
redis-cli -u $REDIS_URL ping
# Should return: PONG
```

**Check 3**: Missing environment variables
```bash
# Compare with example
diff .env .env.example
```

---

### Model Not Loading (503 Service Unavailable)

**Symptom**: All AI endpoints return `503 Service Unavailable`.

**Check 1**: Model files exist
```bash
ls storage/weights/
# Expected: asr/  dialect/  xtts-v2/  translation/
```

**Check 2**: GPU not available
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False: install CUDA toolkit or models will CPU-fallback (much slower)
```

**Check 3**: Out of VRAM
```bash
nvidia-smi
# If VRAM full: reduce concurrent workers or use ONNX instead
```

**Fix**: Models lazy-load on first request. Check backend logs:
```bash
docker logs vietspeech-backend 2>&1 | grep -E "(ERROR|WARNING|loaded)"
```

---

### High ASR Latency (> 300ms)

**Symptoms**: `/api/v1/transcribe` returns `latency_ms > 300`.

**Cause 1**: Model running on CPU
```bash
# Check GPU assignment in logs
grep "model.*cpu\|model.*cuda" backend.log
```

**Cause 2**: Large audio file
- Audio > 30s gets chunked → higher latency
- Use VAD to strip silence: `use_vad=true`

**Cause 3**: No ONNX export
- ONNX inference is 2-3x faster than PyTorch
- Export model: run `training/notebooks/08_export_onnx.ipynb`

---

### Colab Training OOM (Out of Memory)

**Symptom**: `RuntimeError: CUDA out of memory` during training.

**Fix 1**: Reduce batch size in notebook:
```python
# Change in training notebook:
BATCH_SIZE = 2   # was 4
GRAD_ACC = 8     # was 4 (to keep effective batch size)
```

**Fix 2**: Enable QLoRA (4-bit quantization)
```python
# Notebook 03 already uses QLoRA — verify bnb_config:
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
```

**Fix 3**: Clear GPU cache between runs
```python
import torch; torch.cuda.empty_cache()
import gc; gc.collect()
```

**Fix 4**: Use Colab Pro with A100 (40GB) for XTTS-v2 training.

---

### Dialect Accuracy Below Target (< 70%)

**Fix 1**: Check data distribution in training set
```python
# In notebook 02:
print(df_train['province'].value_counts())
# If some provinces have < 50 samples, augment or oversample
```

**Fix 2**: Increase training epochs
```python
NUM_EPOCHS = 50  # was 30
```

**Fix 3**: Unfreeze more layers
```python
# Unfreeze last 6 transformer layers:
for param in model.backbone.encoder.layers[-6:].parameters():
    param.requires_grad = True
```

---

### WER > 8% After Training

**Fix 1**: Ensure audio is 16kHz mono before training
```python
# In dataset prep, check:
print(df['sample_rate'].value_counts())  # should be all 16000
print(df['channels'].value_counts())     # should be all 1
```

**Fix 2**: Increase LoRA rank
```python
lora_config = LoraConfig(
    r=64,          # was 32
    lora_alpha=128,  # was 64
    ...
)
```

**Fix 3**: Use Vietnamese-specific initial prompt
```python
# In transcription:
processor.get_decoder_prompt_ids(language='vi', task='transcribe')
```

---

### Translation Quality Poor

**Expected behavior**: Translation quality depends heavily on corpus size.
The demo corpus (5 pairs) will produce near-random translations.

**Fix**: Obtain the full Ede-Vietnamese corpus:
- Email the IJERT 2015 paper authors
- Check VinAI Research or Vietnamese NLP community resources
- Minimum: 5,000+ pairs for usable translation quality

---

### Docker Compose Volume Permission Error

```bash
# Fix permissions on storage directories
sudo chown -R 1000:1000 storage/
chmod -R 755 storage/
```

---

### Rate Limit Not Working (Too Many 429s)

**Symptom**: Legitimate requests getting rate limited.

**Fix**: Adjust rate limit in `.env`:
```bash
RATE_LIMIT_PER_MINUTE=120  # Default is 60
RATE_LIMIT_BURST=20        # Allow burst of 20
```

Or disable for development:
```bash
DISABLE_RATE_LIMIT=true
```

---

### Flutter App Can't Connect to Backend

**Fix 1**: Check backend URL in Flutter app config
```dart
// lib/core/api/api_client.dart
const String baseUrl = 'http://10.0.2.2:8000'; // Android emulator
// const String baseUrl = 'http://localhost:8000'; // iOS simulator
// const String baseUrl = 'https://your-domain.com'; // Production
```

**Fix 2**: CORS not allowing Flutter's origin
```bash
# Add to .env:
CORS_ORIGINS=http://localhost:8080,http://10.0.2.2:8000
```

---

## Log Locations

| Component | Log Location |
|-----------|-------------|
| Backend | `docker logs vietspeech-backend` |
| Worker | `docker logs vietspeech-worker` |
| Training | MLflow at `http://localhost:5000` |
| Nginx | `/var/log/nginx/error.log` |

---

## Getting Help

1. Check this troubleshooting guide
2. Search GitHub Issues: `github.com/YOUR_USERNAME/vietspeech-ai/issues`
3. Review backend logs: `docker logs vietspeech-backend --tail 100`
4. Open a new GitHub Issue with: error message, logs, OS, GPU info

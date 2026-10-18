# VerifVoice — Launch Guide (team)

> How to run the full system on **your laptop** and on the **LinuxONE VM**.
> Two-minute quick start is at the top; details and troubleshooting are below.

---

## 0. What runs where

| Component | Laptop (Windows) | LinuxONE VM (s390x) |
|---|---|---|
| Coercion rule engine (`semantic_model`) | ✅ | ✅ |
| Semantic classifier (`semantic_classifier`, pure Python) | ✅ | ✅ |
| Risk fusion + token + decision (`risk_adapter`, `verifvoice_risk`, `token_store`) | ✅ | ✅ |
| FastAPI backend (`main.py`) | ✅ | ✅ |
| **Live voice: ASR** (`faster-whisper`) | ✅ | ❌ (no s390x wheel) |
| **Spoof / synthetic-voice** (AASIST ONNX) | ✅ (needs model) | ❌ (no ONNX Runtime s390x wheel) |
| Frontend dashboard (`frontend/index.html`) | ✅ | ✅ (static file) |

The **live-mic "speak and flag"** demo runs on the **laptop**. The VM is the real
"AI runs on LinuxONE" evidence for the **text path** (coercion → fusion → decision).

---

## 1. Quick start — LAPTOP (full demo incl. live voice)

**Prereqs:** Windows + Python 3.12 + git.

```powershell
# 1) get the code
git clone https://github.com/AdityaV1288/VerifVoice_IBM.git
cd VerifVoice_IBM

# 2) virtualenv + deps
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you don't need audio/ASR (text-only demo), a minimal install is enough:
> `pip install fastapi uvicorn pydantic python-multipart`

**Run the backend** (terminal 1):

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**Run the frontend** (terminal 2 — must be HTTP, not double-clicking the file, so the mic works):

```powershell
cd frontend
..\.venv\Scripts\python.exe -m http.server 8080
```

**Open in browser:** 👉 `http://localhost:8080/index.html`

1. Top bar → **Connect** (backend already `http://localhost:8000`) → it should show *"Connected"*.
2. Click a scenario card (e.g. **AI Clone + Coercion**) → **⚡ Analyze Call** → the phone overlay + decision light up.
3. **Live voice:** click **🎙️ Start Live Call**, allow the mic, and speak. It flags coercive speech in ~5 s rolling windows and shows a "Heard" transcript panel.

### 1.1 Make live voice + spoof actually work

- **ASR model (required for mic):** the first audio request auto-downloads
  `faster-whisper-base`. If `huggingface.co` is blocked on your network, pre-download it
  through the mirror and place it under `models/faster-whisper-base/`:

  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  .\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('cached')"
  ```

  (`semantic_model.py` already prefers a local `models/faster-whisper-base/model.bin`.)

- **Spoof / AASIST (optional):** download `models/aasist.onnx`
  (`scripts/download_model.py`), or the demo simply flags `SPOOF_UNAVAILABLE` and rides on
  coercion — which is the primary signal.

---

## 2. LINUXONE VM (s390x) — backend + frontend

**VM:** `linux1@148.100.98.98` (SSH key `VerifVoice.pem`).

```bash
ssh -i VerifVoice.pem linux1@148.100.98.98

# one-time: python env (venv may already exist as ~/verifvoice-env)
python3 -m ensurepip --upgrade          # if pip missing
python3 -m venv ~/verifvoice-env
~/verifvoice-env/bin/pip install fastapi uvicorn pydantic python-multipart

# copy the code (from your laptop)
scp -i VerifVoice.pem -r main.py pipeline.py semantic_model.py semantic_config.py \
  semantic_classifier.py token_store.py risk_adapter.py risk_engine.py \
  verifvoice_risk audio_authenticity frontend models/semantic_classifier.json \
  linux1@148.100.98.98:~/verifvoice/
```

**Start backend** (text path; ASR/spoof are `unavailable` on s390x — expected):

```bash
cd ~/verifvoice
setsid ~/verifvoice-env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 < /dev/null &
```

**Serve frontend:**

```bash
cd ~/verifvoice/frontend
nohup ~/verifvoice-env/bin/python -m http.server 8080 > /tmp/http.log 2>&1 < /dev/null &
```

**Open the firewall** (only SSH is open by default):

```bash
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
```

**Access from your browser:**
- Frontend: `http://148.100.98.98:8080/index.html`
- In the top bar, set the **backend URL** to `http://148.100.98.98:8000` → **Connect**.

**Verify (on the VM):**

```bash
curl http://localhost:8000/health
~/verifvoice-env/bin/python ~/verifvoice/vm_test.py       # full loop
~/verifvoice-env/bin/python ~/verifvoice/vm_evidence.py   # s390x evidence (uname, golden x3, latency)
```

---

## 3. API contract (what the frontend calls)

| Endpoint | Purpose |
|---|---|
| `GET /health` | status + model readiness |
| `POST /analyze` | text (or audio path) → coercion + spoof + token; `?echo_transcript=true` returns the text (demo-only) |
| `POST /analyze/upload` | multipart WAV → ASR + scores + token (for live voice) |
| `POST /payment/decision` | `{token_id, amount, new_beneficiary}` → `observe/warn/verify/pause` |
| `GET /metrics` | latency + request counts |

Decision vocabulary is lowercase **`observe / warn / verify / pause`**.

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `{"detail":"Not Found"}` | You hit the **backend** root `/` (it has no homepage). Open the **frontend** at `http://localhost:8080/index.html`. |
| Mic button does nothing / "needs secure context" | Open the frontend over **http://localhost:8080**, not by double-clicking the file (`file://`). |
| Backend shows *"Unreachable — demo mode"* | Start the backend, then click **Connect**; check the URL field = `http://localhost:8000`. |
| `transcription: unavailable` | ASR model missing — see §1.1. Text path still works. |
| `audio_authenticity: model_unavailable` | AASIST model not present (or you're on s390x). Coercion flag still works; spoof is `SPOOF_UNAVAILABLE`. |
| Port already in use | `Get-NetTCPConnection -LocalPort 8000,8080 -State Listen` → kill the owning PID, or use different ports. |
| VM not reachable externally | Re-run the `iptables -I INPUT` commands in §2 (not persistent across VM reboot). |
| VM backend stopped | Re-run the `setsid … uvicorn` command in §2. |

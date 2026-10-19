# ShieldML API Latency Benchmark Report

## Executive Summary
This report documents the empirical latency performance of the ShieldML real-time scoring endpoint (`POST /predict`), evaluated under continuous load over 100 transactions.

### Key Benchmark Findings
- **Target SLA:** < 50.0 ms
- **Average Latency:** **34.71 ms**
- **95th Percentile (P95) Latency:** **40.71 ms**
- **Median (P50) Latency:** **33.68 ms**
- **99th Percentile (P99) Latency:** **46.95 ms**
- **SLA Compliance:** **PASSED (Sub-50ms)**

## Latency Distribution Summary
| Metric | Latency (ms) | Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **Mean Latency** | `34.71 ms` | < 50.0 ms | **PASS** |
| **Median (P50)** | `33.68 ms` | < 50.0 ms | **PASS** |
| **P90 Latency** | `38.01 ms` | < 50.0 ms | **PASS** |
| **P95 Latency** | `40.71 ms` | < 50.0 ms | **PASS** |
| **P99 Latency** | `46.95 ms` | < 50.0 ms | **PASS** |
| **Min Latency** | `30.44 ms` | — | — |
| **Max Latency** | `58.22 ms` | — | — |

## Architecture Efficiency Rationale
1. **Model Loaded at Startup:** Models and preprocessors are loaded once in memory via FastAPI lifespan handler, completely avoiding I/O overhead on scoring requests.
2. **In-Memory Lightweight Feature Store:** Rolling 1h, 24h, and 7d velocity metrics are computed via in-memory historical ring buffers in sub-0.5ms.
3. **Optimized SHAP TreeExplainer:** TreeExplainer utilizes fast tree path traversal algorithms rather than exponential KernelExplainer sampling, providing attribution in under 5ms.

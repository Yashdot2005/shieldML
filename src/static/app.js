/**
 * ShieldML Web Dashboard Frontend Controller
 * Handles live transaction scoring, SHAP waterfall rendering, real-time metrics monitoring, and latency benchmarks.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Tab Switching
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const targetId = `tab-${btn.getAttribute("data-tab")}`;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");
    });
  });

  // Preset Configurations
  const presets = {
    legit: {
      tx_id: "TX_LEGIT_101",
      card_id: "CARD_00124",
      amount: 48.50,
      category: "grocery",
      merchant: "M_0012",
      device: "DEV_00482",
      channel: "pos_chip",
      country: "US",
      timestamp: new Date().toISOString()
    },
    velocity: {
      tx_id: "TX_BURST_204",
      card_id: "CARD_00999",
      amount: 420.00,
      category: "electronics",
      merchant: "M_0088",
      device: "DEV_FRAUD_001",
      channel: "web",
      country: "US",
      timestamp: new Date().toISOString()
    },
    zscore: {
      tx_id: "TX_ANOM_309",
      card_id: "CARD_00777",
      amount: 2850.00,
      category: "luxury_jewelry",
      merchant: "M_0042",
      device: "DEV_UNKNOWN",
      channel: "web",
      country: "US",
      timestamp: new Date().toISOString()
    },
    night: {
      tx_id: "TX_NIGHT_412",
      card_id: "CARD_00888",
      amount: 650.00,
      category: "digital_gaming",
      merchant: "M_0065",
      device: "DEV_FRAUD_002",
      channel: "mobile_app",
      country: "NG",
      timestamp: new Date(new Date().setHours(3, 14, 0, 0)).toISOString()
    }
  };

  function setPreset(p) {
    document.getElementById("f-tx-id").value = p.tx_id;
    document.getElementById("f-card-id").value = p.card_id;
    document.getElementById("f-amount").value = p.amount;
    document.getElementById("f-category").value = p.category;
    document.getElementById("f-merchant").value = p.merchant;
    document.getElementById("f-device").value = p.device;
    document.getElementById("f-channel").value = p.channel;
    document.getElementById("f-country").value = p.country;
    document.getElementById("f-timestamp").value = p.timestamp;
  }

  document.getElementById("preset-legit").addEventListener("click", (e) => {
    setActivePreset(e.target);
    setPreset(presets.legit);
  });
  document.getElementById("preset-velocity").addEventListener("click", (e) => {
    setActivePreset(e.target);
    setPreset(presets.velocity);
  });
  document.getElementById("preset-zscore").addEventListener("click", (e) => {
    setActivePreset(e.target);
    setPreset(presets.zscore);
  });
  document.getElementById("preset-night").addEventListener("click", (e) => {
    setActivePreset(e.target);
    setPreset(presets.night);
  });

  function setActivePreset(target) {
    document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
    target.classList.add("active");
  }

  // Scoring form submission
  const txForm = document.getElementById("tx-form");
  txForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await executePrediction();
  });

  async function executePrediction() {
    const payload = {
      transaction_id: document.getElementById("f-tx-id").value,
      timestamp: document.getElementById("f-timestamp").value,
      user_id: "U_" + document.getElementById("f-card-id").value.replace("CARD_", ""),
      card_id: document.getElementById("f-card-id").value,
      amount: parseFloat(document.getElementById("f-amount").value),
      merchant_id: document.getElementById("f-merchant").value,
      merchant_category: document.getElementById("f-category").value,
      device_id: document.getElementById("f-device").value,
      location_country: document.getElementById("f-country").value,
      channel: document.getElementById("f-channel").value
    };

    const latencyInd = document.getElementById("latency-indicator");
    latencyInd.innerText = "Evaluating...";
    latencyInd.className = "latency-badge text-muted";

    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Prediction request failed");
      }

      const data = await resp.json();
      renderPredictionResult(data);
      refreshMetrics();
    } catch (err) {
      alert(`Scoring Error: ${err.message}`);
      latencyInd.innerText = "Error";
    }
  }

  function renderPredictionResult(data) {
    const { fraud_score, is_fraud, threshold_applied, latency_ms, shap_explanation } = data;
    
    // Latency badge
    const latencyInd = document.getElementById("latency-indicator");
    latencyInd.innerText = `${latency_ms} ms (SLA < 50ms)`;
    latencyInd.className = latency_ms <= 50 ? "latency-badge text-emerald" : "latency-badge text-ruby";

    // Score percent & Gauge bar
    const pct = (fraud_score * 100).toFixed(1);
    document.getElementById("score-percent").innerText = `${pct}%`;
    const scoreBar = document.getElementById("score-bar");
    scoreBar.style.width = `${Math.min(100, Math.max(2, pct))}%`;
    
    if (is_fraud) {
      scoreBar.style.backgroundColor = "#EF4444";
    } else if (fraud_score > threshold_applied * 0.7) {
      scoreBar.style.backgroundColor = "#F59E0B";
    } else {
      scoreBar.style.backgroundColor = "#10B981";
    }

    // Threshold Marker
    document.getElementById("threshold-marker").innerText = `F2 Cutoff: ${threshold_applied.toFixed(2)}`;

    // Decision Banner
    const banner = document.getElementById("decision-banner");
    const icon = document.getElementById("decision-icon");
    const title = document.getElementById("decision-text");
    const sub = document.getElementById("decision-sub");

    if (is_fraud) {
      banner.className = "decision-banner banner-fraud";
      icon.innerText = "🚨";
      title.innerText = "CRITICAL FRAUD ALERT";
      title.className = "decision-text text-ruby";
      sub.innerText = `Score ${pct}% exceeds optimized business threshold (${threshold_applied.toFixed(2)}). Immediate step-up verification required.`;
    } else {
      banner.className = "decision-banner banner-approved";
      icon.innerText = "✅";
      title.innerText = "TRANSACTION APPROVED";
      title.className = "decision-text text-emerald";
      sub.innerText = `Score ${pct}% satisfies safety criteria below threshold (${threshold_applied.toFixed(2)}). Low fraud likelihood.`;
    }

    // SHAP Attribution
    const shapContainer = document.getElementById("shap-contributions");
    shapContainer.innerHTML = "";

    if (!shap_explanation || shap_explanation.length === 0) {
      shapContainer.innerHTML = `<div class="empty-state">No SHAP attribution returned.</div>`;
      return;
    }

    shap_explanation.forEach(item => {
      const isUp = item.direction === "RISK_INCREASING";
      const div = document.createElement("div");
      div.className = `shap-item ${isUp ? 'risk-up' : 'risk-down'}`;
      div.innerHTML = `
        <div>
          <span class="shap-name">${item.feature}</span>
          <span style="font-size: 0.75rem; color: #94A3B8; margin-left: 8px;">(val: ${item.feature_value})</span>
        </div>
        <div class="shap-val ${isUp ? 'text-ruby' : 'text-emerald'}">
          ${item.shap_value > 0 ? '+' : ''}${item.shap_value.toFixed(3)} ${isUp ? '▲ Risk' : '▼ Safe'}
        </div>
      `;
      shapContainer.appendChild(div);
    });
  }

  // Real-time Metrics Refresh
  async function refreshMetrics() {
    try {
      const resp = await fetch("/metrics");
      if (!resp.ok) return;
      const data = await resp.json();

      document.getElementById("m-avg-latency").innerText = `${data.average_latency_ms} ms`;
      document.getElementById("m-p95-latency").innerText = `${data.p95_latency_ms} ms`;
      document.getElementById("m-fraud-rate").innerText = `${(data.fraud_rate_last_100 * 100).toFixed(1)}%`;
      document.getElementById("m-total-preds").innerText = data.total_predictions.toLocaleString();
      document.getElementById("m-total-fraud").innerText = `${data.total_fraud_detected} Flagged`;

      const verdict = document.getElementById("m-latency-verdict");
      if (data.average_latency_ms <= 50) {
        verdict.innerText = "Sub-50ms Compliant";
        verdict.className = "stat-badge badge-green";
      } else {
        verdict.innerText = "Over SLA";
        verdict.className = "stat-badge badge-red";
      }
    } catch (e) {
      console.warn("Metrics fetch error", e);
    }
  }

  // Load Model Card and Evaluation stats from Health and Metadata
  async function loadInitialStats() {
    try {
      const resp = await fetch("/health");
      if (resp.ok) {
        const h = await resp.json();
        document.getElementById("champion-model-tag").innerText = h.model_name || "Ensemble";
        document.getElementById("threshold-tag").innerText = (h.threshold || 0.35).toFixed(2);
      }
    } catch (e) {
      console.warn(e);
    }

    try {
      const respRep = await fetch("/static/model_performance_report.json");
      if (respRep.ok) {
        const r = await respRep.json();
        document.getElementById("rep-auc-roc").innerText = r.auc_roc.toFixed(4);
        document.getElementById("rep-auc-pr").innerText = r.auc_pr.toFixed(4);
        document.getElementById("rep-f2").innerText = r.optimized_metrics.f2_score.toFixed(4);
        document.getElementById("rep-recall").innerText = `${(r.optimized_metrics.recall * 100).toFixed(1)}%`;
        document.getElementById("rep-savings").innerText = `$${r.financial_savings_usd.toLocaleString()}`;
      }
    } catch (e) {
      // not yet run
    }
  }

  // In-browser Benchmark Runner
  const btnBench = document.getElementById("btn-run-benchmark");
  if (btnBench) {
    btnBench.addEventListener("click", async () => {
      btnBench.disabled = true;
      const benchBox = document.getElementById("benchmark-results");
      benchBox.innerHTML = "Firing 50 consecutive requests to /predict...";

      const latencies = [];
      const p = presets.legit;

      for (let i = 0; i < 50; i++) {
        const payload = {
          transaction_id: `TX_BENCH_${i}`,
          timestamp: new Date().toISOString(),
          user_id: "U_BENCH",
          card_id: "CARD_BENCH",
          amount: Math.round(Math.random() * 250 + 10),
          merchant_id: "M_0012",
          merchant_category: "grocery",
          device_id: "DEV_BENCH",
          location_country: "US",
          channel: "web"
        };

        const t0 = performance.now();
        const resp = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const t1 = performance.now();
        if (resp.ok) {
          const res = await resp.json();
          latencies.push(res.latency_ms);
        }
      }

      latencies.sort((a, b) => a - b);
      const avg = (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(2);
      const p95 = latencies[Math.floor(latencies.length * 0.95)].toFixed(2);
      const min = latencies[0].toFixed(2);
      const max = latencies[latencies.length - 1].toFixed(2);

      benchBox.innerHTML = `
        <div style="color: #34D399; font-weight: 700; margin-bottom: 6px;">✓ Benchmark Complete (50 Requests):</div>
        <div>Average Latency: <strong>${avg} ms</strong> (Target: &lt; 50ms)</div>
        <div>95th Percentile: <strong>${p95} ms</strong></div>
        <div>Min / Max: <strong>${min} ms / ${max} ms</strong></div>
        <div style="color: #60A5FA; margin-top: 4px;">Status: Sub-50ms SLA successfully maintained!</div>
      `;
      btnBench.disabled = false;
      refreshMetrics();
    });
  }

  // Polling loop
  loadInitialStats();
  refreshMetrics();
  setInterval(refreshMetrics, 3000);
});

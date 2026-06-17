/* reward-stack-rl live viewer — vanilla JS, no dependencies. */
(function () {
  "use strict";
  const D = window.RSRL_DATA;
  if (!D) { document.body.innerHTML = "<p style='padding:40px'>data.js not found — run <code>python -m viewer.record</code></p>"; return; }

  const DRIVE_COLORS = { hunger: "#f0a500", safety: "#f85149", social: "#58a6ff", curiosity: "#3fb950", payoff: "#bc8cff" };
  const $ = (id) => document.getElementById(id);
  const lerp = (a, b, t) => a + (b - a) * t;

  /* ---- tabs ---- */
  let activeTab = "grid";
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(btn.dataset.tab).classList.add("active");
      activeTab = btn.dataset.tab;
      if (activeTab === "results") drawResults();
    });
  });

  /* =====================================================================
     ResourceWorld viewer
  ===================================================================== */
  const grid = {
    canvas: $("gridCanvas"), ctx: $("gridCanvas").getContext("2d"),
    variant: "stacked", idx: 0, playing: true, acc: 0, sel: 0,
  };
  const G = D.gridworld;
  const cell = grid.canvas.width / G.W;

  function gridFrames() { return G.variants[grid.variant].frames; }
  function gridDrives() { return G.variants[grid.variant].drives; }

  function drawGrid() {
    const ctx = grid.ctx, frames = gridFrames(), f = frames[grid.idx];
    ctx.clearRect(0, 0, grid.canvas.width, grid.canvas.height);
    // subtle grid
    ctx.strokeStyle = "#161b22"; ctx.lineWidth = 1;
    for (let i = 0; i <= G.W; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, grid.canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(grid.canvas.width, i * cell); ctx.stroke();
    }
    // hazards (static)
    for (const [x, y] of G.hazards) {
      const cx = (x + 0.5) * cell, cy = (y + 0.5) * cell, r = cell * 0.34;
      ctx.fillStyle = "#da3633";
      ctx.beginPath(); ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy + r); ctx.lineTo(cx - r, cy + r);
      ctx.closePath(); ctx.fill();
    }
    // food
    ctx.fillStyle = "#2ea043";
    for (const [x, y] of f.f) {
      ctx.beginPath(); ctx.arc((x + 0.5) * cell, (y + 0.5) * cell, cell * 0.22, 0, 7); ctx.fill();
    }
    // agents
    const drives = gridDrives();
    f.a.forEach((row, i) => {
      const [x, y, dom] = row;
      const cx = (x + 0.5) * cell, cy = (y + 0.5) * cell;
      ctx.beginPath(); ctx.arc(cx, cy, cell * 0.32, 0, 7);
      ctx.fillStyle = "#e6edf3"; ctx.fill();
      ctx.lineWidth = i === grid.sel ? 4 : 3;
      ctx.strokeStyle = DRIVE_COLORS[drives[dom]] || "#888"; ctx.stroke();
      if (i === grid.sel) { ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, cy, cell * 0.44, 0, 7); ctx.stroke(); }
    });
    $("gridFrame").textContent = grid.idx;
    $("gridScrub").value = grid.idx;
    drawDriveBars(f);
  }

  function drawDriveBars(f) {
    const drives = gridDrives(), row = f.a[grid.sel];
    if (!row) return;
    const urg = drives.map((_, k) => row[4 + k]);
    const dom = row[2];
    $("domLabel").textContent = drives[dom];
    $("domLabel").style.color = DRIVE_COLORS[drives[dom]] || "#fff";
    const SCALE = 6.0;
    const html = drives.map((name, k) => {
      const v = urg[k], w = Math.min(100, (v / SCALE) * 100);
      const c = DRIVE_COLORS[name] || "#888";
      return `<div class="bar-row"><span style="color:${c}">${name}</span>`
        + `<div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${c}"></div></div>`
        + `<span class="bar-val">${v.toFixed(2)}</span></div>`;
    }).join("");
    $("driveBars").innerHTML = html;
  }

  function setGridVariant(v) {
    grid.variant = v; grid.idx = 0;
    document.querySelectorAll("#gridVariant button").forEach((b) =>
      b.classList.toggle("on", b.dataset.variant === v));
    $("gridScrub").max = gridFrames().length - 1;
    // agent selector
    const n = gridFrames()[0].a.length;
    if (grid.sel >= n) grid.sel = 0;
    $("agentSel").innerHTML = Array.from({ length: n }, (_, i) => `<option value="${i}">${i}</option>`).join("");
    $("agentSel").value = grid.sel;
    $("gridHint").textContent = v === "stacked"
      ? "Full stack: avoids hazards, clusters with others, wanders when fed — the dominant drive keeps switching."
      : "Survival only: one hunger drive. It just chases food — ignores hazards and company, never switches goals.";
    drawGrid();
  }

  document.querySelectorAll("#gridVariant button").forEach((b) =>
    b.addEventListener("click", () => setGridVariant(b.dataset.variant)));
  $("agentSel").addEventListener("change", (e) => { grid.sel = +e.target.value; drawGrid(); });
  $("gridScrub").addEventListener("input", (e) => { grid.idx = +e.target.value; drawGrid(); });
  $("gridPlay").addEventListener("click", () => {
    grid.playing = !grid.playing; $("gridPlay").textContent = grid.playing ? "⏸ pause" : "▶ play";
  });

  /* =====================================================================
     TugGame viewer
  ===================================================================== */
  const tug = { canvas: $("tugCanvas"), ctx: $("tugCanvas").getContext("2d"),
    variant: "trained", idx: 0, playing: true, acc: 0 };
  const T = D.tug;
  const tscale = tug.canvas.width / T.size;
  const tp = (p) => [p[0] * tscale + tscale / 2, p[1] * tscale + tscale / 2];
  const ANCHOR_COLORS = ["#f0a500", "#58a6ff", "#3fb950"];

  function tugFrames() { return T[tug.variant]; }

  function drawTug() {
    const ctx = tug.ctx, frames = tugFrames(), f = frames[tug.idx];
    ctx.clearRect(0, 0, tug.canvas.width, tug.canvas.height);
    // goal
    const [gx, gy] = tp(T.goal);
    ctx.strokeStyle = "#8b949e"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(gx, gy, tscale * 0.55, 0, 7); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "14px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("B", gx, gy - tscale * 0.7);
    // band lines anchor->center
    const [bx, by] = tp(f.band);
    f.anchors.forEach((a, i) => {
      const [ax, ay] = tp(a);
      ctx.strokeStyle = f.gripped ? "#3fb950" : "#444"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    });
    // bottle
    const [obx, oby] = tp(f.bottle);
    ctx.fillStyle = f.gripped ? "rgba(63,185,80,.25)" : "rgba(248,81,73,.18)";
    ctx.beginPath(); ctx.arc(obx, oby, tscale * 0.42, 0, 7); ctx.fill();
    ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(obx, oby, tscale * 0.42, 0, 7); ctx.stroke();
    ctx.fillStyle = "#e6edf3"; ctx.fillText("O", obx, oby + 4);
    // band center
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(bx, by, 3, 0, 7); ctx.fill();
    // anchors
    f.anchors.forEach((a, i) => {
      const [ax, ay] = tp(a);
      ctx.fillStyle = ANCHOR_COLORS[i]; ctx.beginPath(); ctx.arc(ax, ay, tscale * 0.18, 0, 7); ctx.fill();
    });
    const dist = Math.hypot(f.bottle[0] - T.goal[0], f.bottle[1] - T.goal[1]);
    $("tugGrip").textContent = f.gripped ? "yes" : "no";
    $("tugGrip").style.color = f.gripped ? "#3fb950" : "#f85149";
    $("tugDist").textContent = dist.toFixed(2);
    $("tugFrame").textContent = tug.idx;
    $("tugScrub").value = tug.idx;
  }

  function setTugVariant(v) {
    tug.variant = v; tug.idx = 0;
    document.querySelectorAll("#tugVariant button").forEach((b) =>
      b.classList.toggle("on", b.dataset.variant === v));
    $("tugScrub").max = tugFrames().length - 1;
    drawTug();
  }
  document.querySelectorAll("#tugVariant button").forEach((b) =>
    b.addEventListener("click", () => setTugVariant(b.dataset.variant)));
  $("tugScrub").addEventListener("input", (e) => { tug.idx = +e.target.value; drawTug(); });
  $("tugPlay").addEventListener("click", () => {
    tug.playing = !tug.playing; $("tugPlay").textContent = tug.playing ? "⏸ pause" : "▶ play";
  });

  /* =====================================================================
     animation loop
  ===================================================================== */
  let last = 0;
  function tick(ts) {
    const dt = (ts - last) / 1000; last = ts;
    if (activeTab === "grid" && grid.playing) {
      grid.acc += dt * (+$("gridSpeed").value);
      while (grid.acc >= 1) { grid.acc -= 1; grid.idx = (grid.idx + 1) % gridFrames().length; }
      drawGrid();
    }
    if (activeTab === "tug" && tug.playing) {
      tug.acc += dt * (+$("tugSpeed").value);
      while (tug.acc >= 1) {
        tug.acc -= 1;
        tug.idx++;
        if (tug.idx >= tugFrames().length) tug.idx = 0;  // loop with a pause-at-end feel
      }
      drawTug();
    }
    requestAnimationFrame(tick);
  }

  /* =====================================================================
     Results charts
  ===================================================================== */
  function axes(ctx, w, h, pad) {
    ctx.strokeStyle = "#2d333b"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, 6); ctx.lineTo(pad, h - pad); ctx.lineTo(w - 6, h - pad); ctx.stroke();
  }
  function line(ctx, pts, color, w, h, pad, ymax) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((v, i) => {
      const x = lerp(pad, w - 6, i / (pts.length - 1));
      const y = lerp(h - pad, 6, v / ymax);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  function clearC(c) { const x = c.getContext("2d"); x.clearRect(0, 0, c.width, c.height); return x; }

  let resultsDrawn = false;
  function drawResults() {
    if (resultsDrawn || !D.results) return;
    resultsDrawn = true;
    const R = D.results, pad = 30;

    // 1 — IPD cooperation curves
    if (R.prisoners) {
      const c = $("ipdChart"), ctx = clearC(c); axes(ctx, c.width, c.height, pad);
      line(ctx, R.prisoners.baseline.coop_curve, "#8b949e", c.width, c.height, pad, 1);
      line(ctx, R.prisoners.stacked.coop_curve, "#3fb950", c.width, c.height, pad, 1);
      ctx.fillStyle = "#8b949e"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText("1.0", 6, 12); ctx.fillText("0", 14, c.height - pad + 2);
      ctx.fillStyle = "#3fb950"; ctx.fillText("+reciprocity", pad + 6, 16);
      ctx.fillStyle = "#8b949e"; ctx.fillText("payoff only", pad + 6, 30);
      $("ipdNum").innerHTML = `cooperation <b>${pct(R.prisoners.baseline.final_coop_rate)}</b> → `
        + `<b>${pct(R.prisoners.stacked.final_coop_rate)}</b> &nbsp;·&nbsp; payoff/move `
        + `${R.prisoners.baseline.final_payoff_per_move.toFixed(2)} → ${R.prisoners.stacked.final_payoff_per_move.toFixed(2)}`;
    }

    // 2 — ablation grouped bars (hazard exposure, goal-switch, clustering/0.6)
    if (R.gridworld) {
      const c = $("ablChart"), ctx = clearC(c); axes(ctx, c.width, c.height, pad);
      const cfgs = Object.keys(R.gridworld);
      const series = [
        { key: "hazard_exposure", color: "#f85149", scale: 1, label: "hazard%" },
        { key: "clustering", color: "#58a6ff", scale: 0.6, label: "cluster" },
        { key: "goal_switch_rate", color: "#f0a500", scale: 1, label: "goalsw%" },
      ];
      const groupW = (c.width - pad - 10) / cfgs.length;
      cfgs.forEach((cfg, gi) => {
        const x0 = pad + gi * groupW + 6;
        const bw = (groupW - 12) / series.length;
        series.forEach((s, si) => {
          const v = Math.min(1, R.gridworld[cfg][s.key] / s.scale);
          const bh = (c.height - pad - 6) * v;
          ctx.fillStyle = s.color;
          ctx.fillRect(x0 + si * bw, c.height - pad - bh, bw - 2, bh);
        });
        ctx.fillStyle = "#8b949e"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
        ctx.fillText(cfg.replace(" stack", "stk"), x0 + (groupW - 12) / 2, c.height - pad + 11);
      });
      ctx.textAlign = "left";
      series.forEach((s, i) => { ctx.fillStyle = s.color; ctx.fillText(s.label, pad + 6 + i * 70, 14); });
      $("ablNum").innerHTML = `safety: hazard <b>${pct(R.gridworld.hunger.hazard_exposure)}</b>→`
        + `<b>${pct(R.gridworld["+safety"].hazard_exposure)}</b> · social: cluster `
        + `<b>${R.gridworld.hunger.clustering.toFixed(2)}</b>→<b>${R.gridworld["+social"].clustering.toFixed(2)}</b>`
        + ` · stack goal-switch <b>${pct(R.gridworld["full stack"].goal_switch_rate)}</b>`;
    }

    // 3 — tug delivery pre/post
    if (R.tug) {
      const c = $("tugChart"), ctx = clearC(c); axes(ctx, c.width, c.height, pad);
      const bars = [
        { l: "before", v: R.tug.plain.pre_delivery_rate, color: "#f85149" },
        { l: "after", v: R.tug.plain.post_delivery_rate, color: "#3fb950" },
      ];
      bars.forEach((b, i) => {
        const x = pad + 40 + i * 150, bw = 90, bh = (c.height - pad - 10) * b.v;
        ctx.fillStyle = b.color; ctx.fillRect(x, c.height - pad - bh, bw, bh);
        ctx.fillStyle = "#c9d1d9"; ctx.font = "12px sans-serif"; ctx.textAlign = "center";
        ctx.fillText(pct(b.v), x + bw / 2, c.height - pad - bh - 6);
        ctx.fillStyle = "#8b949e"; ctx.fillText(b.l, x + bw / 2, c.height - pad + 13);
      });
      $("tugNum").innerHTML = `delivery <b>${pct(R.tug.plain.pre_delivery_rate)}</b> → `
        + `<b>${pct(R.tug.plain.post_delivery_rate)}</b> &nbsp;·&nbsp; steps `
        + `${R.tug.plain.pre_mean_steps.toFixed(0)} → ${R.tug.plain.post_mean_steps.toFixed(0)}`;
    }

    // 4 — evolution gene drift
    if (R.evolve) {
      const c = $("evoChart"), ctx = clearC(c); axes(ctx, c.width, c.height, pad);
      const genes = R.evolve.history.map((h) => h.gene_means.reciprocity);
      const ymax = Math.max(...genes) * 1.1;
      line(ctx, genes, "#bc8cff", c.width, c.height, pad, ymax);
      ctx.fillStyle = "#8b949e"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(ymax.toFixed(1), 6, 12); ctx.fillText("gen 0", pad, c.height - pad + 13);
      ctx.fillStyle = "#bc8cff"; ctx.fillText("reciprocity gene", pad + 6, 16);
      $("evoNum").innerHTML = `gene <b>${genes[0].toFixed(2)}</b> → <b>${R.evolve.best_reciprocity.toFixed(2)}</b>`
        + ` &nbsp;·&nbsp; cooperation (by-product) <b>${pct(R.evolve.coop_seed)}</b> → <b>${pct(R.evolve.coop_evolved)}</b>`;
    }
  }
  const pct = (x) => Math.round(x * 100) + "%";

  /* ---- boot ---- */
  setGridVariant("stacked");
  setTugVariant("trained");
  requestAnimationFrame(tick);
})();

(function () {
  const $ = (s) => document.querySelector(s);
  const SAMPLES = {
    short: "Says the state legislature has cut school funding every single year for the past decade.",
    sensational:
      "BREAKING!!! You will NOT BELIEVE what officials in the town of Riverbend are HIDING from you!!! Insiders say a SHOCKING secret plan will DESTROY the local economy and nobody in the media is willing to tell the TRUTH!!! " +
      "This is a DISGRACE and a total BETRAYAL of every honest family in the county!!! Our sources, who must remain secret, claim the council has secretly been lying for YEARS and that the scandal will EXPOSE a massive cover-up!!! " +
      "WAKE UP, people!!! They do NOT want you to read this!!! Share it NOW before it gets BANNED!!! The corrupt elite are PANICKING because the evil truth is finally out!!! " +
      "Experts WARN of a terrible disaster, a horrible collapse, and a deadly crisis that could RUIN everything you love!!! Do not trust the so-called official reports, they are fake, fraudulent and a hoax!!! " +
      "Thousands are already ANGRY and the outrage is spreading FAST!!! This is the WORST thing that has ever happened and the authorities are doing NOTHING!!! " + "Remember, you heard it here FIRST!!! ".repeat(6),
    measured:
      "The Riverbend town council voted on Tuesday to approve a revised budget for the coming fiscal year. According to the published agenda, the plan allocates funds to road maintenance, the public library, and a small increase in spending for the fire department. " +
      "Council members discussed the proposal for roughly two hours before the vote. Several residents spoke during the public comment period, and some asked for more detail about how the library hours would change. The finance director said the figures were reviewed by an outside auditor last month and that a summary will be posted on the town website. " +
      "The vote was five to two. One member who voted against the plan said she supported the goals but wanted a longer review period, while another said he was concerned about the projected cost of road repairs. The mayor said the council would revisit the numbers at a meeting in the spring if revenue came in lower than expected. " +
      "The budget takes effect on the first of July. Residents can read the full document at the town hall or online, and the next public meeting is scheduled for the second Tuesday of next month. Officials said they welcome questions and will answer them during that session or by email. ".repeat(2),
  };

  // ---- tabs
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
      $("#mode-text").hidden = t.dataset.mode !== "text";
      $("#mode-sliders").hidden = t.dataset.mode !== "sliders";
    })
  );
  document.querySelectorAll("[data-sample]").forEach((b) =>
    b.addEventListener("click", () => { $("#txt").value = SAMPLES[b.dataset.sample]; count(); })
  );
  const count = () => {
    const n = ($("#txt").value.match(/[A-Za-z0-9][A-Za-z0-9'’-]*/g) || []).length;
    $("#count").textContent = n + " words";
  };
  $("#txt").addEventListener("input", count);
  [["wc", "o_wc"], ["ss", "o_ss"], ["ex", "o_ex"], ["cp", "o_cp"]].forEach(([i, o]) =>
    $("#" + i).addEventListener("input", () => ($("#" + o).textContent = $("#" + i).value))
  );

  // ---- metrics
  try {
    const m = JSON.parse($("#metrics-data").textContent || "{}");
    const box = $("#metrics");
    const add = (k, v) => { const d = document.createElement("div"); d.className = "metric"; d.innerHTML = "<span class='muted small'></span><b></b>"; d.firstChild.textContent = k; d.lastChild.textContent = v; box.appendChild(d); };
    if (m.text) { add("Text model accuracy (LIAR test)", (m.text.test.accuracy * 100).toFixed(1) + "%"); add("Text model macro-F1", m.text.test.macro_f1.toFixed(3)); }
    if (m.style) { add("Style ensemble accuracy (hold-out)", (m.style.test.accuracy * 100).toFixed(1) + "%"); add("Style ensemble F1 (FAKE)", m.style.test.f1.toFixed(3)); }
  } catch (e) { /* metrics are optional */ }

  // ---- calling the API
  async function call(payload, btn) {
    btn.disabled = true; const old = btn.textContent; btn.textContent = "Analysing...";
    try {
      const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Request failed");
      render(j);
    } catch (e) { alert(e.message); }
    finally { btn.disabled = false; btn.textContent = old; }
  }
  $("#go").addEventListener("click", () => call({ text: $("#txt").value }, $("#go")));
  $("#go2").addEventListener("click", () =>
    call({ features: { word_count: +$("#wc").value, sentiment_score: +$("#ss").value, exclamation_count: +$("#ex").value, all_caps_pct: +$("#cp").value } }, $("#go2"))
  );

  const bars = (el, rows, scale) => {
    el.innerHTML = "";
    rows.forEach((r) => {
      const w = Math.min(50, Math.abs(r.v) / scale * 50);
      const d = document.createElement("div"); d.className = "bar";
      d.innerHTML = "<span></span><div class='track'><div class='fill'></div></div><span></span>";
      d.children[0].textContent = r.name;
      const f = d.querySelector(".fill"); f.className = "fill " + (r.v >= 0 ? "pos" : "neg"); f.style.width = w + "%";
      d.children[2].textContent = (r.v >= 0 ? "+" : "") + r.v.toFixed(3);
      el.appendChild(d);
    });
    if (!rows.length) el.textContent = "No explanation available.";
  };

  function render(j) {
    $("#result").hidden = false;
    const v = $("#verdict"); v.textContent = j.verdict;
    v.className = "verdict " + (j.verdict.startsWith("LIKELY FAKE") ? "fake" : j.verdict.startsWith("LIKELY REAL") ? "real" : "unc");
    $("#needle").style.left = (j.p_fake * 100).toFixed(1) + "%";
    $("#pf").textContent = "P(fake-style) = " + (j.p_fake * 100).toFixed(0) + "%";
    const w = $("#warnings"); w.innerHTML = "";
    (j.warnings || []).forEach((t) => { const d = document.createElement("div"); d.className = "warn"; d.textContent = t; w.appendChild(d); });
    const ft = $("#ftable"); ft.innerHTML = "";
    Object.entries(j.features).forEach(([k, val]) => { const tr = ft.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = val; });
    if (j.style_model) { $("#exstyle").hidden = false; bars($("#bars_style"), j.style_model.explanation.items.map((i) => ({ name: i.label, v: i.contribution })), 0.15); } else $("#exstyle").hidden = true;
    if (j.text_model) { $("#extext").hidden = false; bars($("#bars_text"), j.text_model.explanation.items.map((i) => ({ name: i.word, v: i.weight })), 0.08); } else $("#extext").hidden = true;
    const parts = [];
    if (j.style_model) parts.push("style model P(fake) = " + j.style_model.p_fake);
    if (j.text_model) parts.push("text model P(fake) = " + j.text_model.p_fake);
    $("#foot").textContent = parts.join(" | ") + " | " + j.latency_ms + " ms. " + j.disclaimer;
    $("#result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  count();
})();

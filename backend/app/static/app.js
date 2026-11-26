async function api(path, options) {
  const resp = await fetch(path, options);
  const data = await resp.json();
  if (!resp.ok) {
    const msg = typeof data.detail === "string" ? data.detail : "ошибка запроса";
    throw new Error(msg);
  }
  return data;
}

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function percent(x) {
  return (x * 100).toFixed(1) + "%";
}

function showError(el, err) {
  el.innerHTML = '<p class="error">' + esc(err.message) + "</p>";
}

async function predict() {
  const out = document.getElementById("predict-result");
  const text = document.getElementById("review").value;
  const showAll = document.getElementById("show-all").checked;
  out.textContent = "считаю...";
  try {
    const data = await api("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text}),
    });
    const rows = showAll ? data.scores : data.scores.slice(0, 7);
    let html = "<p>Найдено: <b>" + (data.labels.length ? esc(data.labels.join(", ")) : "ничего") + "</b></p>";
    html += "<table><tr><th>Характеристика</th><th>Группа</th><th>Вероятность</th><th>Порог</th><th style='width:30%'></th></tr>";
    for (const s of rows) {
      html += "<tr class='" + (s.predicted ? "yes" : "") + "'><td>" + esc(s.label) + "</td><td>" + esc(s.group) +
        "</td><td class='num'>" + percent(s.probability) + "</td><td class='num'>" + s.threshold.toFixed(2) +
        "</td><td><div class='bar' style='width:" + (s.probability * 100).toFixed(0) + "%'></div></td></tr>";
    }
    html += "</table><p class='muted'>Жирным - характеристики, у которых вероятность не ниже порога. " +
      "Порог подобран для каждой характеристики отдельно.</p>";
    out.innerHTML = html;
  } catch (err) {
    showError(out, err);
  }
}

function init() {
  // ссылка вида /?text=... сразу показывает результат
  const params = new URLSearchParams(location.search);
  if (params.get("text")) {
    document.getElementById("review").value = params.get("text");
    predict();
  }
}

document.getElementById("btn-predict").addEventListener("click", predict);
init();

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

function venuesTable(items, withShare) {
  if (!items.length) return "<p>Ничего не найдено</p>";
  let html = "<table><tr><th>Заведение</th><th>Тип</th><th>Рейтинг</th><th>Отзывов</th>" +
    (withShare ? "<th>Доля отзывов</th>" : "") + "</tr>";
  for (const v of items) {
    html += "<tr><td>" + esc(v.name) + "</td><td>" + esc(v.kind || "") + "</td><td class='num'>" +
      (v.rating == null ? "" : v.rating) + "</td><td class='num'>" + v.reviews + "</td>" +
      (withShare ? "<td class='num'>" + percent(v.share) + "</td>" : "") + "</tr>";
  }
  return html + "</table>";
}

async function loadTop() {
  const out = document.getElementById("top-result");
  const label = document.getElementById("label-select").value;
  const minReviews = document.getElementById("min-reviews").value || 10;
  try {
    const params = new URLSearchParams({label: label, limit: 10, min_reviews: minReviews});
    out.innerHTML = venuesTable(await api("/api/venues?" + params), true);
  } catch (err) {
    showError(out, err);
  }
}

async function searchVenues() {
  const out = document.getElementById("search-result");
  const q = document.getElementById("venue-query").value.trim();
  document.getElementById("venue-result").innerHTML = "";
  if (q.length < 2) {
    out.innerHTML = "<p>Введите хотя бы 2 буквы</p>";
    return;
  }
  try {
    const items = await api("/api/venues/search?q=" + encodeURIComponent(q));
    if (!items.length) {
      out.innerHTML = "<p>Ничего не найдено</p>";
      return;
    }
    out.innerHTML = items.map(v =>
      "<div class='found'><a data-id='" + esc(v.branch_id) + "'>" + esc(v.name) + "</a> " +
      esc(v.kind || "") + ", отзывов: " + v.reviews + "</div>").join("");
    out.querySelectorAll("a").forEach(a => a.addEventListener("click", () => showVenue(a.dataset.id)));
  } catch (err) {
    showError(out, err);
  }
}

async function showVenue(id) {
  const out = document.getElementById("venue-result");
  try {
    const v = await api("/api/venues/" + encodeURIComponent(id));
    let html = "<h3>" + esc(v.name) + (v.kind ? ", " + esc(v.kind) : "") + "</h3>";
    html += "<p>Рейтинг: " + (v.rating == null ? "-" : v.rating) + ", отзывов в выборке: " + v.reviews + "</p>";
    html += "<table><tr><th>Характеристика</th><th>Группа</th><th>Отзывов</th><th>Доля</th></tr>";
    for (const l of v.labels) {
      html += "<tr><td>" + esc(l.label) + "</td><td>" + esc(l.group) + "</td><td class='num'>" + l.count +
        "</td><td class='num'>" + percent(l.share) + "</td></tr>";
    }
    out.innerHTML = html + "</table>";
  } catch (err) {
    showError(out, err);
  }
}

async function init() {
  try {
    const health = await api("/api/health");
    document.getElementById("model-name").textContent = health.model === "bert" ? "ruBERT" : "TF-IDF + логистическая регрессия";
    const labels = await api("/api/labels");
    const select = document.getElementById("label-select");
    for (const l of labels) {
      const opt = document.createElement("option");
      opt.value = l.label;
      opt.textContent = l.label;
      if (l.label === "Свидание") opt.selected = true;
      select.appendChild(opt);
    }
  } catch (err) {
    document.getElementById("model-name").textContent = "не загружена";
    return;
  }

  // ссылки вида /?text=...&label=Свидание сразу показывают результат
  const params = new URLSearchParams(location.search);
  if (params.get("text")) {
    document.getElementById("review").value = params.get("text");
    predict();
  }
  if (params.get("label")) {
    document.getElementById("label-select").value = params.get("label");
    loadTop();
  }
}

document.getElementById("btn-predict").addEventListener("click", predict);
document.getElementById("btn-top").addEventListener("click", loadTop);
document.getElementById("btn-search").addEventListener("click", searchVenues);
document.getElementById("venue-query").addEventListener("keydown", e => {
  if (e.key === "Enter") searchVenues();
});
init();

"""Minimal zero-dependency web server for the routing service.

GET  /                -> single-page UI
GET  /api/configs     -> available router configs
GET  /api/history     -> recent routing decisions (?limit=N)
POST /api/route       -> {prompt, config, task?} -> decision (and records it)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from xrouter_llm.serving import RoutingService


def create_handler(service: RoutingService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "xrouter-llm/0.1"

        def log_message(self, *args: object) -> None:  # quieter logs
            return

        def _send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
            elif parsed.path == "/api/configs":
                self._send_json(
                    {"configs": [c.to_dict() for c in service.configs.values()]}
                )
            elif parsed.path == "/api/history":
                limit = int((parse_qs(parsed.query).get("limit", ["50"]))[0])
                self._send_json({"calls": service.store.recent(limit)})
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/route":
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = service.route(
                    str(payload.get("prompt", "")),
                    config_name=str(payload.get("config", "")),
                    task=payload.get("task") or None,
                )
            except (KeyError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:  # surface anything else as 500
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return
            self._send_json(result)

    return Handler


def run_server(service: RoutingService, *, host: str = "127.0.0.1", port: int = 8080) -> None:
    httpd = ThreadingHTTPServer((host, port), create_handler(service))
    print(f"xrouter-llm serving on http://{host}:{port}  (configs: {', '.join(service.configs)})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>xrouter-llm</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  header { padding: 16px 24px; border-bottom: 1px solid #2a2f3a; }
  h1 { font-size: 18px; margin: 0; } h1 small { color: #8a93a6; font-weight: normal; }
  main { max-width: 1000px; margin: 0 auto; padding: 24px; }
  .card { background: #171a21; border: 1px solid #2a2f3a; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
  label { display: block; font-size: 12px; color: #8a93a6; margin: 8px 0 4px; }
  textarea, select { width: 100%; box-sizing: border-box; background: #0f1115; color: #e6e6e6;
    border: 1px solid #2a2f3a; border-radius: 8px; padding: 10px; font: inherit; }
  textarea { min-height: 90px; resize: vertical; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
  .row > div { flex: 1; min-width: 160px; }
  button { background: #3b82f6; color: white; border: 0; border-radius: 8px; padding: 10px 18px;
    font: inherit; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #232834; vertical-align: top; }
  th { color: #8a93a6; font-weight: 600; }
  .pick { color: #4ade80; font-weight: 700; }
  .muted { color: #8a93a6; } .mono { font-family: ui-monospace, monospace; }
  .bar { height: 6px; background: #232834; border-radius: 3px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #3b82f6; }
  .err { color: #f87171; }
</style>
</head>
<body>
<header><h1>xrouter-llm <small>routing decision service</small></h1></header>
<main>
  <div class="card">
    <div class="row">
      <div style="flex:3"><label>Prompt</label><textarea id="prompt" placeholder="Paste a prompt..."></textarea></div>
    </div>
    <div class="row">
      <div><label>Auto config</label><select id="config"></select></div>
      <div style="flex:0 0 auto"><label>&nbsp;</label><button id="go">Route</button></div>
    </div>
    <div id="cfgInfo" class="muted" style="margin-top:8px;font-size:12px;"></div>
  </div>

  <div class="card" id="resultCard" style="display:none">
    <h3 style="margin:0 0 10px">Decision</h3>
    <div id="result"></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h3 style="margin:0">History</h3>
      <button id="refresh" style="background:#2a2f3a">Refresh</button>
    </div>
    <div id="history" style="margin-top:10px"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let CONFIGS = {};

async function loadConfigs() {
  const r = await fetch('/api/configs'); const d = await r.json();
  CONFIGS = {}; const sel = $('config'); sel.innerHTML = '';
  d.configs.forEach(c => { CONFIGS[c.name] = c; const o = document.createElement('option'); o.value = c.name; o.textContent = c.name + ' (' + c.models.length + ' model' + (c.models.length>1?'s':'') + ')'; sel.appendChild(o); });
  showCfg();
}
function showCfg() {
  const c = CONFIGS[$('config').value]; if (!c) return;
  $('cfgInfo').textContent = (c.description ? c.description + ' · ' : '') + 'threshold ' + c.completion_threshold + ' · models: ' + c.models.join(', ');
}
function fmtCost(x){ return '$' + Number(x).toFixed(6); }
function pct(x){ return (Number(x)*100).toFixed(1) + '%'; }

async function route() {
  $('go').disabled = true;
  try {
    const r = await fetch('/api/route', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({prompt: $('prompt').value, config: $('config').value})});
    const d = await r.json();
    const card = $('resultCard'); card.style.display = 'block';
    if (d.error) { $('result').innerHTML = '<div class="err">'+d.error+'</div>'; return; }
    let rows = d.candidates.map(c => {
      const picked = d.selected.includes(c.model_id);
      return '<tr><td class="'+(picked?'pick':'')+'">'+(picked?'→ ':'')+c.model_id+'</td>'+
        '<td><div class="bar"><i style="width:'+(c.mu*100)+'%"></i></div>'+pct(c.mu)+'</td>'+
        '<td class="muted">±'+c.sigma.toFixed(3)+'</td>'+
        '<td class="mono">'+fmtCost(c.cost)+'</td></tr>';
    }).join('');
    $('result').innerHTML = '<p>Selected <span class="pick mono">'+d.selected.join(' + ')+
      '</span> &middot; expected completion '+pct(d.expected_quality)+' &middot; cost '+fmtCost(d.cost)+'</p>'+
      '<table><thead><tr><th>model</th><th>predicted completion</th><th>uncertainty</th><th>est. cost</th></tr></thead><tbody>'+rows+'</tbody></table>';
    loadHistory();
  } finally { $('go').disabled = false; }
}

async function loadHistory() {
  const r = await fetch('/api/history?limit=50'); const d = await r.json();
  if (!d.calls.length) { $('history').innerHTML = '<span class="muted">No calls yet.</span>'; return; }
  const rows = d.calls.map(c => '<tr><td class="muted mono">#'+c.id+'</td>'+
    '<td class="muted">'+new Date(c.ts*1000).toLocaleString()+'</td>'+
    '<td>'+c.config+'</td>'+
    '<td title="'+(c.prompt||'').replace(/"/g,'&quot;')+'">'+(c.prompt||'').slice(0,48)+((c.prompt||'').length>48?'…':'')+'</td>'+
    '<td class="pick">'+c.selected.join(' + ')+'</td>'+
    '<td>'+pct(c.expected_quality)+'</td>'+
    '<td class="mono">'+fmtCost(c.cost)+'</td></tr>').join('');
  $('history').innerHTML = '<table><thead><tr><th>#</th><th>time</th><th>config</th><th>prompt</th><th>selected</th><th>exp.</th><th>cost</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

$('go').onclick = route;
$('refresh').onclick = loadHistory;
$('config').onchange = showCfg;
loadConfigs(); loadHistory();
</script>
</body>
</html>"""

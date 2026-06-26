"""FastAPI server for the routing decision service.

Endpoints
---------
GET  /                -> single-page UI
GET  /api/configs     -> available preset router configs
GET  /api/history     -> recent routing decisions (?limit=N)
POST /api/route       -> routing decision (and records it)

Xinference Cloud integration
----------------------------
Mount the router into an existing FastAPI app:

    from xrouter_llm.server import create_router
    app.include_router(create_router(service), prefix="/xrouter")
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from xrouter_llm.serving import RoutingService


class RouteRequest(BaseModel):
    prompt: str
    models: list[str] | None = Field(
        default=None,
        description="Candidate model IDs. Omit to route across all registered models.",
    )
    task: str | None = None
    completion_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    lambda_cost: float = Field(default=1.0, ge=0.0)
    lambda_latency: float = Field(default=0.0, ge=0.0)
    max_k: int = Field(default=1, ge=1)
    fallback_quality_margin: float = Field(default=0.05, ge=0.0, le=1.0)


def create_router(service: RoutingService) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @router.get("/api/configs")
    def get_configs() -> dict[str, Any]:
        return {"configs": [c.to_dict() for c in service.configs.values()]}

    @router.get("/api/history")
    def get_history(limit: int = 50) -> dict[str, Any]:
        return {"calls": service.store.recent(limit)}

    @router.post("/api/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        try:
            return service.route(
                req.prompt,
                models=req.models,
                task=req.task,
                completion_threshold=req.completion_threshold,
                lambda_cost=req.lambda_cost,
                lambda_latency=req.lambda_latency,
                max_k=req.max_k,
                fallback_quality_margin=req.fallback_quality_margin,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


def create_app(service: RoutingService) -> FastAPI:
    app = FastAPI(
        title="xrouter-llm",
        description="LLM routing decision service",
    )
    app.include_router(create_router(service))
    return app


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
  textarea, input[type=text], input[type=number] { width: 100%; box-sizing: border-box; background: #0f1115; color: #e6e6e6;
    border: 1px solid #2a2f3a; border-radius: 8px; padding: 10px; font: inherit; }
  textarea { min-height: 90px; resize: vertical; }
  input[type=number] { width: auto; max-width: 120px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
  .row > div { flex: 1; min-width: 160px; }
  button { background: #3b82f6; color: white; border: 0; border-radius: 8px; padding: 10px 18px;
    font: inherit; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  button.secondary { background: #2a2f3a; }
  summary { cursor: pointer; color: #8a93a6; font-size: 12px; margin-top: 8px; user-select: none; }
  details > div { margin-top: 10px; }
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
      <div style="flex:3">
        <label>Models <span class="muted">(comma-separated, leave empty for all)</span></label>
        <input type="text" id="models" placeholder="e.g. anthropic/claude-sonnet-4.6, deepseek/deepseek-v4-flash">
      </div>
    </div>
    <details>
      <summary>Advanced params</summary>
      <div class="row">
        <div><label>completion_threshold</label><input type="number" id="threshold" value="0.7" min="0" max="1" step="0.05"></div>
        <div><label>lambda_cost</label><input type="number" id="lambda_cost" value="1.0" min="0" step="0.1"></div>
        <div><label>lambda_latency</label><input type="number" id="lambda_latency" value="0.0" min="0" step="0.1"></div>
        <div><label>max_k</label><input type="number" id="max_k" value="1" min="1" step="1"></div>
        <div><label>fallback_quality_margin</label><input type="number" id="margin" value="0.05" min="0" max="1" step="0.01"></div>
      </div>
    </details>
    <div class="row" style="margin-top:12px">
      <div style="flex:0 0 auto"><button id="go">Route</button></div>
    </div>
  </div>

  <div class="card" id="resultCard" style="display:none">
    <h3 style="margin:0 0 10px">Decision</h3>
    <div id="result"></div>
  </div>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h3 style="margin:0">History</h3>
      <button id="refresh" class="secondary">Refresh</button>
    </div>
    <div id="history" style="margin-top:10px"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
function fmtCost(x){ return '$' + Number(x).toFixed(6); }
function pct(x){ return (Number(x)*100).toFixed(1) + '%'; }

async function route() {
  $('go').disabled = true;
  try {
    const modelsRaw = $('models').value.trim();
    const body = {
      prompt: $('prompt').value,
      completion_threshold: parseFloat($('threshold').value),
      lambda_cost: parseFloat($('lambda_cost').value),
      lambda_latency: parseFloat($('lambda_latency').value),
      max_k: parseInt($('max_k').value),
      fallback_quality_margin: parseFloat($('margin').value),
    };
    if (modelsRaw) body.models = modelsRaw.split(',').map(s => s.trim()).filter(Boolean);
    const r = await fetch('/api/route', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const d = await r.json();
    const card = $('resultCard'); card.style.display = 'block';
    if (d.detail) { $('result').innerHTML = '<div class="err">'+d.detail+'</div>'; return; }
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
    '<td title="'+(c.prompt||'').replace(/"/g,'&quot;')+'">'+(c.prompt||'').slice(0,48)+((c.prompt||'').length>48?'…':'')+'</td>'+
    '<td class="pick">'+c.selected.join(' + ')+'</td>'+
    '<td>'+pct(c.expected_quality)+'</td>'+
    '<td class="mono">'+fmtCost(c.cost)+'</td></tr>').join('');
  $('history').innerHTML = '<table><thead><tr><th>#</th><th>time</th><th>prompt</th><th>selected</th><th>exp.</th><th>cost</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

$('go').onclick = route;
$('refresh').onclick = loadHistory;
loadHistory();
</script>
</body>
</html>"""

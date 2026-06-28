"""FastAPI server for the routing decision service.

Endpoints
---------
GET  /                          -> single-page UI
GET  /api/configs               -> available preset router configs
GET  /api/history               -> recent routing decisions (?limit=N&offset=M)
POST /api/route                 -> routing decision (and records it)
PATCH /api/calls/{id}/feedback  -> record user feedback on a routing decision
DELETE /api/calls/{id}          -> delete a call record

Xinference Cloud integration
----------------------------
Mount the router into an existing FastAPI app:

    from xrouter_llm.server import create_router
    app.include_router(create_router(service), prefix="/xrouter")
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from xrouter_llm.serving import RoutingService


class RouteRequest(BaseModel):
    prompt: str
    config_name: str | None = Field(
        default=None,
        description="Named router config. Its models and policy are used as defaults; "
                    "any explicit field below overrides the config value.",
    )
    config: str | None = Field(
        default=None,
        description="Alias for config_name (legacy field name).",
    )
    models: list[str] | None = Field(
        default=None,
        description="Candidate model IDs. Omit to use config models or all registered models.",
    )
    task: str | None = None
    user_id: str | None = Field(default=None, max_length=255, description="Caller identity; used for per-user history filtering.")
    completion_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    lambda_cost: float | None = Field(default=None, ge=0.0)
    lambda_latency: float | None = Field(default=None, ge=0.0)
    max_k: int | None = Field(default=None, ge=1)
    fallback_quality_margin: float | None = Field(default=None, ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    outcome: Literal["good", "bad", "retracted"] = Field(description="'good', 'bad', or 'retracted' to clear.")
    correct_model: str | None = Field(default=None, min_length=1, max_length=200, description="Model that should have been selected.")
    note: str | None = Field(default=None, min_length=1, max_length=1000, description="Free-text comment.")

    @model_validator(mode="after")
    def _validate_feedback_fields(self) -> "FeedbackRequest":
        if self.outcome != "bad" and self.correct_model is not None:
            raise ValueError("correct_model can only be specified when outcome is 'bad'")
        if self.outcome == "retracted" and self.note is not None:
            raise ValueError("note cannot be specified when outcome is 'retracted'")
        return self


def create_router(service: RoutingService) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @router.get("/api/configs")
    def get_configs() -> dict[str, Any]:
        return {"configs": [c.to_dict() for c in service.configs.values()]}

    @router.get("/api/history")
    def get_history(limit: int = 20, offset: int = 0, user_id: str | None = None) -> dict[str, Any]:
        return {
            "calls": service.store.recent(limit, offset, user_id=user_id),
            "total": service.store.count(user_id=user_id),
        }

    @router.post("/api/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        try:
            return service.route(
                req.prompt,
                config_name=req.config_name or req.config,
                models=req.models,
                task=req.task,
                completion_threshold=req.completion_threshold,
                lambda_cost=req.lambda_cost,
                lambda_latency=req.lambda_latency,
                max_k=req.max_k,
                fallback_quality_margin=req.fallback_quality_margin,
                user_id=req.user_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.patch("/api/calls/{call_id}/feedback")
    def set_feedback(call_id: int, body: FeedbackRequest) -> dict[str, Any]:
        feedback: dict[str, Any] | None = None
        if body.outcome != "retracted":
            feedback = {"outcome": body.outcome}
            if body.correct_model is not None:
                feedback["correct_model"] = body.correct_model
            if body.note is not None:
                feedback["note"] = body.note
        if not service.store.set_feedback(call_id, feedback):
            raise HTTPException(status_code=404, detail=f"call {call_id} not found")
        return {"id": call_id, "feedback": feedback}

    @router.delete("/api/calls/{call_id}")
    def delete_call(call_id: int) -> dict[str, Any]:
        if not service.store.delete(call_id):
            raise HTTPException(status_code=404, detail=f"call {call_id} not found")
        return {"deleted": call_id}

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
  button.secondary { background: #2a2f3a; font-weight: normal; }
  summary { cursor: pointer; color: #8a93a6; font-size: 12px; margin-top: 8px; user-select: none; }
  details > div { margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #232834; vertical-align: middle; }
  th { color: #8a93a6; font-weight: 600; }
  .pick { color: #4ade80; font-weight: 700; }
  .muted { color: #8a93a6; } .mono { font-family: ui-monospace, monospace; }
  .bar { height: 6px; background: #232834; border-radius: 3px; overflow: hidden; display: inline-block; width: 60px; vertical-align: middle; margin-right: 4px; }
  .bar > i { display: block; height: 100%; background: #3b82f6; }
  .err { color: #f87171; }
  /* history table */
  .hist-table tr.main-row:hover > td { background: #1a1e27; }
  .expand-btn { background: none; border: none; color: #8a93a6; cursor: pointer; padding: 0 4px;
    font-size: 11px; vertical-align: middle; border-radius: 3px; }
  .expand-btn:hover { color: #e6e6e6; background: #2a2f3a; }
  .detail-row > td { padding: 0; border-bottom: 2px solid #2a2f3a; }
  .detail-inner { padding: 12px 16px 16px; background: #0f1115; }
  .full-prompt { font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap;
    word-break: break-all; color: #c8cdd8; margin-bottom: 12px;
    max-height: 180px; overflow-y: auto; line-height: 1.6; }
  .cand-table { font-size: 12px; }
  .cand-table th, .cand-table td { padding: 4px 8px; border-bottom: 1px solid #1a1e27; }
  .fb-btn { background: none; border: none; cursor: pointer; font-size: 15px; padding: 2px 4px; border-radius: 4px; line-height: 1; }
  .fb-btn:hover { background: #2a2f3a; }
  .fb-btn.active-good { color: #4ade80; }
  .fb-btn.active-bad  { color: #f87171; }
  .fb-cell { display: flex; align-items: center; gap: 2px; }
  .del-btn { background: none; border: none; color: #4a5568; cursor: pointer;
    padding: 2px 7px; border-radius: 4px; font-size: 14px; line-height: 1; }
  .del-btn:hover { background: #2d1515; color: #f87171; }
  .del-confirm { display: flex; align-items: center; gap: 4px; font-size: 12px; color: #f87171; white-space: nowrap; }
  .del-confirm button { padding: 2px 7px; font-size: 12px; font-weight: normal; border-radius: 4px; }
  .del-confirm .yes { background: #7f1d1d; }
  .del-confirm .no  { background: #2a2f3a; }
  /* pagination */
  .hist-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .hist-header h3 { margin: 0; }
  .pager { display: flex; align-items: center; gap: 8px; }
  .pager span { color: #8a93a6; font-size: 13px; }
  .pager button { padding: 5px 12px; font-size: 14px; font-weight: normal; }
  .prompt-cell { max-width: 260px; }
  .prompt-short { display: inline; }
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
    <div class="hist-header">
      <h3>History</h3>
      <div class="pager">
        <span id="pageInfo"></span>
        <button class="secondary" id="prevPage" onclick="changePage(-1)">&#8249;</button>
        <button class="secondary" id="nextPage" onclick="changePage(1)">&#8250;</button>
        <button class="secondary" id="refresh" onclick="loadHistory()">Refresh</button>
      </div>
    </div>
    <div id="history"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const PAGE_SIZE = 20;
let _page = 0, _total = 0;

function fmtCost(x){ return '$' + Number(x).toFixed(6); }
function pct(x){ return (Number(x)*100).toFixed(1) + '%'; }
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

/* ── routing ── */
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
    $('resultCard').style.display = 'block';
    if (d.detail) { $('result').innerHTML = '<div class="err">'+esc(d.detail)+'</div>'; return; }
    $('result').innerHTML =
      '<p>Selected <span class="pick mono">'+esc(d.selected.join(' + '))+
      '</span> &middot; completion '+pct(d.expected_quality)+' &middot; cost '+fmtCost(d.cost)+'</p>'+
      candidatesTable(d.candidates, d.selected);
    _page = 0; loadHistory();
  } finally { $('go').disabled = false; }
}

function candidatesTable(candidates, selected) {
  const rows = candidates.map(c => {
    const picked = selected.includes(c.model_id);
    return '<tr><td class="'+(picked?'pick':'')+'">'+(picked?'→ ':'')+esc(c.model_id)+'</td>'+
      '<td><span class="bar"><i style="width:'+(c.mu*100)+'%"></i></span>'+pct(c.mu)+'</td>'+
      '<td class="muted">&#177;'+c.sigma.toFixed(3)+'</td>'+
      '<td class="mono">'+fmtCost(c.cost)+'</td></tr>';
  }).join('');
  return '<table><thead><tr><th>model</th><th>predicted completion</th><th>&#963;</th><th>est. cost</th></tr></thead><tbody>'+rows+'</tbody></table>';
}

/* ── history ── */
async function loadHistory() {
  const offset = _page * PAGE_SIZE;
  const r = await fetch('/api/history?limit='+PAGE_SIZE+'&offset='+offset);
  const d = await r.json();
  _total = d.total;
  renderHistory(d.calls);
  renderPager();
}

function renderPager() {
  const start = _total ? _page * PAGE_SIZE + 1 : 0;
  const end = Math.min((_page + 1) * PAGE_SIZE, _total);
  $('pageInfo').textContent = _total ? start+'–'+end+' / '+_total : '暂无记录';
  $('prevPage').disabled = _page === 0;
  $('nextPage').disabled = end >= _total;
}

function changePage(dir) {
  _page = Math.max(0, _page + dir);
  loadHistory();
}

function renderHistory(calls) {
  if (!calls.length) {
    $('history').innerHTML = '<span class="muted">No calls yet.</span>';
    return;
  }
  let tbody = '';
  for (const c of calls) {
    const short = c.prompt.length > 60 ? esc(c.prompt.slice(0,60))+'…' : esc(c.prompt);
    const needExpand = c.prompt.length > 60 || (c.candidates||[]).length;
    tbody +=
      '<tr class="main-row" data-id="'+c.id+'">'+
        '<td class="muted mono">#'+c.id+'</td>'+
        '<td class="muted" style="white-space:nowrap">'+esc(new Date(c.ts*1000).toLocaleString())+'</td>'+
        '<td class="prompt-cell">'+
          '<span class="prompt-short">'+short+'</span>'+
          (needExpand ? '<button class="expand-btn" onclick="toggleDetail('+c.id+',this)">&#9660;</button>' : '')+
        '</td>'+
        '<td class="pick">'+esc((c.selected||[]).join(' + '))+'</td>'+
        '<td>'+pct(c.expected_quality)+'</td>'+
        '<td class="mono">'+fmtCost(c.cost)+'</td>'+
        '<td><div class="fb-cell" id="fb-'+c.id+'">'+fbButtons(c)+'</div></td>'+
        '<td><button class="del-btn" id="del-'+c.id+'" onclick="startDelete('+c.id+')">&#128465;</button></td>'+
      '</tr>'+
      '<tr class="detail-row" id="detail-'+c.id+'" style="display:none">'+
        '<td colspan="8">'+
          '<div class="detail-inner">'+
            '<div class="full-prompt">'+esc(c.prompt)+'</div>'+
            ((c.candidates||[]).length ? candidatesTable(c.candidates, c.selected||[]).replace('<table>','<table class="cand-table">') : '')+
          '</div>'+
        '</td>'+
      '</tr>';
  }
  $('history').innerHTML =
    '<table class="hist-table"><thead><tr>'+
      '<th>#</th><th>Time</th><th>Prompt</th><th>Selected</th><th>Exp.</th><th>Cost</th><th></th><th></th>'+
    '</tr></thead><tbody>'+tbody+'</tbody></table>';
}

/* ── feedback ── */
function fbButtons(c) {
  const fb = c.feedback || {};
  const good = fb.outcome === 'good' ? ' active-good' : '';
  const bad  = fb.outcome === 'bad'  ? ' active-bad'  : '';
  return '<button class="fb-btn'+good+'" title="Good routing" onclick="sendFeedback('+c.id+',1,this)">👍</button>'+
         '<button class="fb-btn'+bad+'"  title="Bad routing"  onclick="sendFeedback('+c.id+',0,this)">👎</button>';
}

async function sendFeedback(id, isGood, btn) {
  const cell = btn.closest('.fb-cell');
  if (cell.dataset.loading) return;
  cell.dataset.loading = 'true';
  const outcome = isGood ? 'good' : 'bad';
  const isToggle = btn.classList.contains(isGood ? 'active-good' : 'active-bad');
  const newOutcome = isToggle ? 'retracted' : outcome;
  try {
    const r = await fetch('/api/calls/'+id+'/feedback', {
      method: 'PATCH',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({outcome: newOutcome}),
    });
    if (r.ok) {
      const fakeCall = {id, feedback: newOutcome === 'retracted' ? null : {outcome: newOutcome}};
      cell.innerHTML = fbButtons(fakeCall);
    } else {
      alert('Failed to update feedback: ' + r.statusText);
    }
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    delete cell.dataset.loading;
  }
}

function toggleDetail(id, btn) {
  const row = $('detail-'+id);
  const hidden = row.style.display === 'none';
  row.style.display = hidden ? '' : 'none';
  btn.innerHTML = hidden ? '&#9650;' : '&#9660;';
}

/* ── delete ── */
function startDelete(id) {
  const cell = $('del-'+id);
  cell.outerHTML =
    '<td><div class="del-confirm" id="del-'+id+'">'+
      '删除?'+
      '<button class="yes" onclick="doDelete('+id+')">&#10003;</button>'+
      '<button class="no"  onclick="cancelDelete('+id+',this)">&#10005;</button>'+
    '</div></td>';
}

function cancelDelete(id, btn) {
  const cell = btn.closest('td');
  cell.outerHTML = '<td><button class="del-btn" id="del-'+id+'" onclick="startDelete('+id+')">&#128465;</button></td>';
}

async function doDelete(id) {
  const r = await fetch('/api/calls/'+id, {method:'DELETE'});
  if (!r.ok) return;
  document.querySelector('tr[data-id="'+id+'"]')?.remove();
  $('detail-'+id)?.remove();
  _total = Math.max(0, _total - 1);
  renderPager();
  if (!document.querySelectorAll('.main-row').length) {
    if (_page > 0) { _page--; } loadHistory();
  }
}

$('go').onclick = route;
loadHistory();
</script>
</body>
</html>"""

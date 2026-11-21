import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import Plot from 'react-plotly.js';
import {
  Activity, ArrowRight, BarChart3, Check, CheckCheck, ChevronRight, CircleAlert, Clock3,
  Code2, Copy, Cpu, Database, Download, FileText, GitBranch, Layers3, Lightbulb, Menu,
  MessageSquare, Play, Plus, RefreshCw, Send, ShieldCheck, Sparkles, Table2, Terminal,
  TrendingDown, TrendingUp, X,
} from 'lucide-react';
import { api } from './api';
import type { HealthResponse } from './api';
import { DEMO_SHOWCASES } from './demoRuns';
import type { DemoShowcase } from './demoRuns';
import type {
  Chart, ChatResponse, Evidence, ModelUsage, PythonAnalysisItem,
  Report, TaskGraphItem, TraceEvent, Turn,
} from './types';

const PROMPTS = [
  { icon: TrendingDown, tag: 'Root Cause Analysis', text: 'Why did sales decline in November 2004 compared with October 2004?', note: 'Identify key drivers across product lines, regions, and order volume' },
  { icon: BarChart3, tag: 'Top Performers', text: 'Show the top 10 products by revenue in 2004', note: 'Rank best-selling products with revenue breakdown and visual comparison' },
  { icon: TrendingUp, tag: 'Year-over-Year Comparison', text: 'Compare 2004 vs 2003 revenue by product line', note: 'Analyze annual revenue growth, category shifts, and performance trends' },
  { icon: FileText, tag: 'Executive Summary', text: 'Generate a comprehensive executive business report on November 2004 revenue performance', note: 'Full executive overview with key metrics, findings, and PDF export' },
];

const PROGRESS_STEPS = ['Understanding question', 'Gathering business data', 'Analyzing trends & drivers', 'Preparing charts & insights'];
const currencyFormatter = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
const numberFormatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

function displayNumber(value: number | null, unit: string) {
  if (value == null) return '—';
  if (unit === 'USD') return currencyFormatter.format(value);
  if (unit === '%') return `${value > 0 ? '+' : ''}${numberFormatter.format(value)}%`;
  return unit ? `${numberFormatter.format(value)} ${unit}` : numberFormatter.format(value);
}

const ACRONYM_MAP: Record<string, string> = {
  aov: 'AOV',
  ytd: 'YTD',
  mom: 'MoM',
  yoy: 'YoY',
  kpi: 'KPI',
  kpis: 'KPIs',
  usd: 'USD',
  rfm: 'RFM',
  sql: 'SQL',
  id: 'ID',
  pct: '%',
};

function formatLabel(raw?: string): string {
  if (!raw) return '';
  return raw
    .replaceAll('_', ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map(w => ACRONYM_MAP[w.toLowerCase()] || (w === w.toUpperCase() && w.length <= 5 ? w : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()))
    .join(' ');
}

function stripMarkdown(raw?: string): string {
  return raw ? raw.replace(/\*\*(.+?)\*\*/g, '$1').replace(/__(.+?)__/g, '$1').replace(/`(.+?)`/g, '$1').replace(/^#{1,6}\s+/gm, '').trim() : '';
}

function renderInlineFormatting(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, idx) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={idx}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={idx}>{part.replace(/`([^`]+)`/g, '$1')}</span>
    )
  );
}

function RichText({ text }: { text?: string }) {
  if (!text) return null;
  const blocks: ReactNode[] = [];
  let listItems: string[] = [];
  const flushList = (keyPrefix: string) => {
    if (listItems.length > 0) {
      blocks.push(<ul key={`list-${keyPrefix}`}>{listItems.map((item, i) => <li key={i}>{renderInlineFormatting(item)}</li>)}</ul>);
      listItems = [];
    }
  };
  text.split(/\r?\n/).forEach((rawLine, idx) => {
    const trimmed = rawLine.trim().replace(/^#{1,6}\s+/, '');
    if (!trimmed) return flushList(String(idx));
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
      listItems.push(trimmed.slice(2).trim());
    } else {
      flushList(String(idx));
      blocks.push(<p key={`p-${idx}`}>{renderInlineFormatting(trimmed)}</p>);
    }
  });
  flushList('end');
  return <div className="rich-text">{blocks}</div>;
}

function resolveProgressStepIndex(events: TraceEvent[]): number {
  if (!events.length) return 0;
  const latest = (events[events.length - 1]?.stage || '').toLowerCase();
  if (['insight', 'synthesize', 'report', 'chart', 'quality'].some(k => latest.includes(k))) return 3;
  if (['python', 'hypothesis', 'review', 'aggregate', 'refinement'].some(k => latest.includes(k))) return 2;
  if (['sql', 'exec', 'supervisor'].some(k => latest.includes(k))) return 1;
  return 0;
}

function openDemoReportWindow(response: ChatResponse, autoPrint = false) {
  const r = response.report;
  const win = window.open('', '_blank');
  if (!win) return;
  const kpiCards = (r.kpis || []).map(k => `<div class="kpi"><div class="lbl">${k.label}</div><div class="val">${displayNumber(k.value, k.unit)}</div></div>`).join('');
  const sectionsOrFindings = r.business_report?.sections?.length
    ? r.business_report.sections.map((s, i) => `<div class="card"><strong>${i + 1}. ${stripMarkdown(s.heading)}</strong><p>${stripMarkdown(s.body)}</p></div>`).join('')
    : (r.findings || []).map(f => `<div class="card"><strong>${stripMarkdown(f.title)}</strong><p>${stripMarkdown(f.detail)}</p></div>`).join('');
  const recsList = (r.recommendations || []).map((rec, i) => `<li><strong>${i + 1}.</strong> ${stripMarkdown(rec)}</li>`).join('');
  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"/><title>${stripMarkdown(r.title)}</title>
  <style>
    body { font-family: Inter, -apple-system, sans-serif; color: #0f172a; background: #f8fafc; margin: 0; padding: 28px; line-height: 1.6; }
    .page { max-width: 900px; margin: 0 auto; background: #fff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 20px rgba(15,23,42,0.05); }
    .hero { border-top: 4px solid #059669; border-bottom: 1px solid #e2e8f0; padding: 28px 36px 22px; }
    .hero h1 { margin: 0 0 6px; font-size: 24px; color: #0f172a; } .hero small { color: #64748b; font-size: 12.5px; }
    .body { padding: 26px 36px 32px; }
    .summary { background: #f8faf9; border: 1px solid #d1fae5; border-left: 4px solid #059669; padding: 16px 20px; border-radius: 8px; margin-bottom: 24px; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .kpi { background: #fff; border: 1px solid #e2e8f0; padding: 14px; border-radius: 8px; }
    .kpi .lbl { font-size: 11.5px; color: #64748b; } .kpi .val { font-size: 20px; font-weight: 700; margin-top: 4px; color: #0f172a; }
    .card { border: 1px solid #e2e8f0; border-left: 3px solid #059669; padding: 14px 16px; border-radius: 8px; margin-bottom: 12px; }
    .card p { margin: 6px 0 0; font-size: 13.5px; color: #334155; }
    h3 { color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 24px; font-size: 15.5px; }
    ul { padding-left: 18px; color: #334155; font-size: 13.5px; }
  </style></head><body><div class="page"><div class="hero"><h1>${stripMarkdown(r.title)}</h1>
  <small>Analysis Period: ${r.period}${r.comparison ? ` vs. ${r.comparison}` : ''}</small></div>
  <div class="body"><div class="summary"><strong>Executive Summary</strong><p>${stripMarkdown(r.summary)}</p></div>
  <div class="kpis">${kpiCards}</div><h3>Analysis &amp; Findings</h3>${sectionsOrFindings}${recsList ? `<h3>Recommendations</h3><ul>${recsList}</ul>` : ''}</div></div></body></html>`);
  win.document.close();
  if (autoPrint) setTimeout(() => win.print(), 250);
}

function DataChart({ chart }: { chart: Chart }) {
  const [chartMode, setChartMode] = useState<'bar' | 'line'>(chart.kind === 'line' ? 'line' : 'bar');
  useEffect(() => setChartMode(chart.kind === 'line' ? 'line' : 'bar'), [chart.kind, chart.title]);

  const isLine = chartMode === 'line';
  const labels = Array.isArray(chart.labels) ? chart.labels : [];
  const values = Array.isArray(chart.values) ? chart.values.map(v => Number(v) || 0) : [];
  const isCurrency = chart.unit === 'USD';
  const isPercent = chart.unit === '%';

  let peakLabel = '';
  let peakVal = 0;
  if (values.length > 0) {
    let bestIdx = 0;
    for (let i = 1; i < values.length; i++) if (Math.abs(values[i]) > Math.abs(values[bestIdx])) bestIdx = i;
    peakLabel = labels[bestIdx] || '';
    peakVal = values[bestIdx] || 0;
  }

  const valueTemplate = isCurrency ? '$%{y:,.2f}' : isPercent ? '%{y:,.2f}%' : '%{y:,.2f}';
  const barValueTemplate = isCurrency ? '$%{x:,.2f}' : isPercent ? '%{x:,.2f}%' : '%{x:,.2f}';

  const data = isLine
    ? {
        type: 'scatter' as const, mode: 'lines+markers' as const, x: labels, y: values,
        line: { color: '#059669', width: 3, shape: 'spline' as const }, marker: { color: '#059669', size: 7 },
        fill: 'tozeroy' as const, fillcolor: 'rgba(16, 185, 129, 0.11)',
        hovertemplate: `%{x}<br>${valueTemplate}<extra></extra>`,
      }
    : {
        type: 'bar' as const, orientation: 'h' as const, y: labels, x: values,
        text: values.map(v => displayNumber(v, chart.unit)), textposition: 'auto' as const, insidetextanchor: 'end' as const,
        marker: { color: values.map(v => (v < 0 ? '#dc2626' : '#059669')), line: { width: 0 } },
        cornerradius: 5, hovertemplate: `%{y}<br>${barValueTemplate}<extra></extra>`,
      };

  const chartHeight = isLine ? 310 : Math.max(280, Math.min(440, labels.length * 32 + 90));

  return (
    <div className="chart-panel">
      <div className="panel-heading">
        <div>
          <div className="panel-heading-title">{chart.title}</div>
          {peakLabel && (
            <div className="panel-heading-sub">
              <span>Primary contributor: <strong>{peakLabel}</strong> ({displayNumber(peakVal, chart.unit)})</span>
            </div>
          )}
        </div>
        <div className="chart-type-toggle" role="group" aria-label="Chart style">
          <button type="button" className={!isLine ? 'active' : ''} onClick={() => setChartMode('bar')}>Bar</button>
          <button type="button" className={isLine ? 'active' : ''} onClick={() => setChartMode('line')}>Trend</button>
        </div>
      </div>
      <Plot
        data={[data]}
        layout={{
          autosize: true, height: chartHeight, margin: { l: isLine ? 58 : 175, r: 28, t: 12, b: 48 },
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
          font: { family: 'Inter, system-ui, sans-serif', color: '#475569', size: 11.5 },
          xaxis: { gridcolor: '#f1f5f3', zerolinecolor: '#94a3b8', tickformat: isLine ? undefined : isCurrency ? '$,.2s' : undefined, automargin: true },
          yaxis: { gridcolor: isLine ? '#f1f5f3' : 'transparent', automargin: true, autorange: isLine ? undefined : 'reversed' },
          hoverlabel: { bgcolor: '#065f46', font: { color: '#ffffff', size: 12 } },
        }}
        style={{ width: '100%', height: chartHeight }}
        useResizeHandler
        config={{ displayModeBar: false, responsive: true }}
      />
    </div>
  );
}

function downloadTableCsv(title: string, rows: Array<Record<string, unknown>>) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0] || {});
  const escapeCsv = (val: unknown) => {
    const s = String(val ?? '');
    return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const csv = [cols.map(escapeCsv).join(','), ...rows.map(r => cols.map(c => escapeCsv(r[c])).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

const COUNT_TOKENS = ['count', 'orders', 'total_orders', 'purchasing_customers', 'customers', 'units', 'qty', 'quantity', 'year', 'cluster', 'days', 'frequency', 'score', 'index'];
const PCT_TOKENS = ['pct', 'percent', 'percentage', 'share', 'rate'];

function formatTableCell(val: unknown, colKey: string, row: Record<string, unknown>): string {
  if (val == null || val === '') return '—';
  if (typeof val === 'string') {
    const s = val.trim();
    if (s.includes('_') && !s.startsWith('http') && !s.startsWith('ds_') && !s.startsWith('q_')) {
      return formatLabel(s);
    }
    return s;
  }
  if (typeof val !== 'number') return String(val);

  const lk = colKey.toLowerCase();
  const isChange = ['change', 'variance', 'delta'].some(t => lk.includes(t));
  if (PCT_TOKENS.some(t => lk.includes(t))) {
    return `${isChange && val > 0 ? '+' : ''}${numberFormatter.format(val)}%`;
  }

  let rowMetric = '';
  for (const mk of ['metric', 'measure', 'kpi', 'check', 'indicator', 'item', 'variable']) {
    if (typeof row[mk] === 'string') {
      rowMetric = String(row[mk]).toLowerCase();
      break;
    }
  }

  if (rowMetric && ['current', 'baseline', 'previous', 'change', 'variance', 'delta', 'value', 'diff'].some(t => lk.includes(t))) {
    if (PCT_TOKENS.some(t => rowMetric.includes(t))) {
      return `${isChange && val > 0 ? '+' : ''}${numberFormatter.format(val)}%`;
    }
    if (COUNT_TOKENS.some(t => rowMetric.includes(t))) {
      return `${isChange && val > 0 ? '+' : ''}${numberFormatter.format(val)}`;
    }
    return `${isChange && val > 0 ? '+' : ''}${currencyFormatter.format(val)}`;
  }

  if (COUNT_TOKENS.some(t => lk.includes(t))) {
    return `${isChange && val > 0 ? '+' : ''}${numberFormatter.format(val)}`;
  }
  return `${isChange && val > 0 ? '+' : ''}${currencyFormatter.format(val)}`;
}

function FormattedDataTable({ title, rows, style }: { title: string; rows: Array<Record<string, unknown>>; style?: React.CSSProperties }) {
  const cols = Object.keys(rows[0] || {});
  const isNumericCol = (col: string) => rows.some(r => typeof r[col] === 'number');

  return (
    <section className="data-table" style={style}>
      <div className="data-table-header">
        <h3><Table2 size={16} /> {title}</h3>
        <div className="table-header-actions">
          <small>{rows.length} rows</small>
          <button type="button" className="csv-btn" onClick={() => downloadTableCsv(title, rows)}><Download size={12} /> CSV</button>
        </div>
      </div>
      <div className="table-scroller">
        <table>
          <thead>
            <tr>{cols.map(k => <th key={k} className={isNumericCol(k) ? 'num' : ''}>{formatLabel(k)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, j) => (
              <tr key={j}>
                {cols.map(key => {
                  const val = row[key];
                  const lk = key.toLowerCase();
                  const isNum = typeof val === 'number';
                  const isVar = isNum && ['change', 'variance', 'pct', 'delta'].some(t => lk.includes(t));
                  const tone = isVar ? ((val as number) < 0 ? 'neg-val' : (val as number) > 0 ? 'pos-val' : '') : '';
                  const formatted = formatTableCell(val, key, row);
                  return <td key={key} className={`${isNum ? 'num' : ''} ${tone}`.trim()}>{formatted}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CopyCodeButton({ text, label = 'Copy Code' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="copy-code-btn"
      onClick={() => {
        navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1800);
      }}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? 'Copied' : label}
    </button>
  );
}

function ProvenanceView({ response }: { response: ChatResponse }) {
  const report = response.report;
  const pythonAnalyses: PythonAnalysisItem[] = report.python_analyses || [];
  const evidenceList: Evidence[] = report.evidence || [];
  const modelsUsed: ModelUsage[] = report.models_used || [];
  const taskGraph: TaskGraphItem[] = report.task_graph || [];
  const traceEvents: TraceEvent[] = report.trace || [];
  const review = report.evidence_review;

  const totalTokens = modelsUsed.reduce((acc, m) => acc + (Number(m.tokens) || 0), 0);
  const totalCost = modelsUsed.reduce((acc, m) => acc + (Number(m.cost_usd) || 0), 0);
  const totalModelMs = modelsUsed.reduce((acc, m) => acc + (Number(m.duration_ms) || 0), 0);

  return (
    <div className="provenance-stack">
      <section className="report-section provenance-banner">
        <div className="provenance-banner-top">
          <div className="provenance-title-group">
            <ShieldCheck size={20} />
            <div>
              <h4>3-Layer Independent Verification & Guardrail Audit</h4>
              <p>
                Every SQL query is AST-validated via SQLGlot (`ReadOnlyGateway`), Python scripts execute in an isolated
                child process with `sys.addaudithook`, and numerical claims are independently reproduced against SHA-256 dataset snapshots.
              </p>
            </div>
          </div>
          <span className="state-pill good">
            <CheckCheck size={13} />{' '}
            {review?.decision === 'sufficient' || report.status === 'complete' ? '3-Layer Audit Passed' : formatLabel(review?.decision || report.status)}
          </span>
        </div>

        <div className="provenance-metrics-row">
          <div className="prov-metric">
            <span>Jev Decision Route</span>
            <strong>{response.route}</strong>
            <small>{Math.round((response.route_confidence || 1) * 100)}% confidence</small>
          </div>
          <div className="prov-metric">
            <span>Verified SQL Snapshots</span>
            <strong>{evidenceList.length} Queries</strong>
            <small>Read-only MySQL AST verified</small>
          </div>
          <div className="prov-metric">
            <span>Python Sandbox Runs</span>
            <strong>{pythonAnalyses.length} Executed</strong>
            <small>{pythonAnalyses.some(p => p.verification_status === 'reproduced') ? '100% SHA-256 Reproduced' : 'Isolated Process Sandbox'}</small>
          </div>
          <div className="prov-metric">
            <span>Multi-Model Token Usage</span>
            <strong>{totalTokens.toLocaleString()} tokens</strong>
            <small>{totalCost > 0 ? `$${totalCost.toFixed(4)} est. cost` : `${modelsUsed.length} model calls`}</small>
          </div>
        </div>

        {review?.notes && review.notes.length > 0 && (
          <div className="review-notes-box">
            {review.notes.map((note, idx) => (
              <div key={idx} className="review-note-item"><Check size={14} /><span>{note}</span></div>
            ))}
          </div>
        )}
      </section>

      {taskGraph.length > 0 && (
        <section className="report-section">
          <h3><GitBranch size={17} /> LangGraph Supervisor-Worker Task DAG ({taskGraph.length} Tasks)</h3>
          <div className="dag-grid">
            {taskGraph.map(task => (
              <div key={task.id} className="dag-card">
                <div className="dag-card-header">
                  <span className={`agent-badge agent-${task.agent}`}>
                    {task.agent === 'sql' ? 'SQL Agent' : task.agent === 'python' ? 'Python Sandbox' : 'Hypothesis Agent'}
                  </span>
                  <span className={`state-pill ${task.status === 'COMPLETED' ? 'good' : 'warn'}`}>{task.status}</span>
                </div>
                <strong className="dag-task-title">{task.title}</strong>
                <div className="dag-task-meta">
                  <code>id: {task.id}</code>
                  <span>{task.duration_ms} ms</span>
                  {task.depends_on?.length > 0 && <span>depends on: {task.depends_on.join(', ')}</span>}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {pythonAnalyses.length > 0 && (
        <section className="report-section">
          <h3><Terminal size={17} /> Autonomous Python Sandbox & SHA-256 Reproduction ({pythonAnalyses.length})</h3>
          <div className="prov-card-list">
            {pythonAnalyses.map((py, idx) => {
              const method = py.methodology || py.calculation?.methodology || formatLabel(py.tool);
              const isReproduced = py.verification_status === 'reproduced' || py.verification_status === 'verified';
              const summaryEntries = Object.entries(py.calculation?.summary || {}).slice(0, 10);

              return (
                <div key={idx} className="prov-code-card">
                  <div className="prov-code-header">
                    <div>
                      <div className="prov-code-badges">
                        <span className="agent-badge agent-python"><Code2 size={12} /> Task: {py.task_id}</span>
                        <span className={`state-pill ${isReproduced ? 'good' : 'warn'}`}>
                          <ShieldCheck size={12} />{' '}
                          {isReproduced ? 'SHA-256 Sandbox Reproduced' : py.verification_status ? formatLabel(py.verification_status) : 'Deterministic Tool Verified'}
                        </span>
                        <span className="meta-chip">{py.duration_ms} ms</span>
                        {typeof py.repair_attempts === 'number' && <span className="meta-chip">Self-repairs: {py.repair_attempts}</span>}
                      </div>
                      <h4 className="prov-method-title">{method}</h4>
                    </div>
                    {py.generated_code && <CopyCodeButton text={py.generated_code} label="Copy Python" />}
                  </div>

                  {py.assumptions_checked && py.assumptions_checked.length > 0 && (
                    <div className="assumptions-box">
                      <strong>Statistical & Accounting Assumptions Verified:</strong>
                      <ul>{py.assumptions_checked.map((a, i) => <li key={i}>{a}</li>)}</ul>
                    </div>
                  )}

                  {summaryEntries.length > 0 && (
                    <div className="calc-summary-chips">
                      {summaryEntries.map(([k, v]) => (
                        <div key={k} className="calc-chip">
                          <span>{formatLabel(k)}</span>
                          <strong>
                            {typeof v === 'number'
                              ? k.includes('usd') || k.includes('revenue') || k.includes('variance')
                                ? currencyFormatter.format(v)
                                : numberFormatter.format(v)
                              : String(v)}
                          </strong>
                        </div>
                      ))}
                    </div>
                  )}

                  {py.execution_steps && py.execution_steps.length > 0 && (
                    <div className="loop-steps-box">
                      <div className="loop-steps-title">Autonomous Agent Loop (`Understand → Plan → Act → Observe → Reflect → Finish`)</div>
                      {py.execution_steps.map((st, sIdx) => (
                        <div key={sIdx} className="loop-step-row">
                          <span className="loop-step-num">Step {st.step_index || sIdx + 1}</span>
                          <div className="loop-step-body">
                            <div className="loop-step-top">
                              <strong>{formatLabel(st.phase)} · <code>{st.action}</code></strong>
                              {st.duration_ms !== undefined && <small>{st.duration_ms} ms</small>}
                            </div>
                            {st.thought && <p>{st.thought}</p>}
                            {st.observation_summary && <pre className="loop-obs-pre">{st.observation_summary}</pre>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {py.generated_code && <pre className="code-block"><code>{py.generated_code}</code></pre>}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {evidenceList.length > 0 && (
        <section className="report-section">
          <h3><Database size={17} /> Executed Read-Only SQL Queries & SHA-256 Snapshots ({evidenceList.length})</h3>
          <div className="prov-card-list">
            {evidenceList.map((ev, idx) => (
              <div key={idx} className="prov-code-card">
                <div className="prov-code-header">
                  <div>
                    <div className="prov-code-badges">
                      <span className="agent-badge agent-sql">SQL Task: {ev.task_id}</span>
                      <span className="state-pill good">SQLGlot AST Read-Only Verified</span>
                      <span className="meta-chip">{ev.rows?.length || 0} rows</span>
                      <span className="meta-chip">{ev.duration_ms} ms</span>
                      {ev.dataset_id && <span className="meta-chip">Snapshot: {ev.dataset_id}</span>}
                    </div>
                    <h4 className="prov-method-title">{ev.purpose || `Query ${ev.query_id}`}</h4>
                  </div>
                  {ev.sql && <CopyCodeButton text={ev.sql} label="Copy SQL" />}
                </div>
                {ev.sql && <pre className="code-block sql-block"><code>{ev.sql}</code></pre>}
              </div>
            ))}
          </div>
        </section>
      )}

      {modelsUsed.length > 0 && (
        <section className="data-table">
          <div className="data-table-header">
            <h3><Cpu size={16} /> Multi-Model Workload Routing & Cost Telemetry</h3>
            <div className="table-header-actions">
              <small>{modelsUsed.length} calls · {totalTokens.toLocaleString()} tokens · {(totalModelMs / 1000).toFixed(2)}s cumulative LLM time</small>
            </div>
          </div>
          <div className="table-scroller">
            <table>
              <thead>
                <tr>
                  <th>Agent Role</th><th>Task Workload</th><th>Assigned Model</th><th>Reasoning Effort</th>
                  <th className="num">Latency</th><th className="num">Tokens</th><th className="num">Est. Cost (USD)</th>
                </tr>
              </thead>
              <tbody>
                {modelsUsed.map((m, idx) => (
                  <tr key={idx}>
                    <td><strong>{m.agent_role}</strong></td>
                    <td>{formatLabel(m.task_type)}</td>
                    <td><code>{m.model}</code></td>
                    <td>{formatLabel(m.reasoning_effort)}</td>
                    <td className="num">{m.duration_ms.toLocaleString()} ms</td>
                    <td className="num">{(m.tokens || 0).toLocaleString()}</td>
                    <td className="num">{m.cost_usd ? `$${Number(m.cost_usd).toFixed(5)}` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {traceEvents.length > 0 && (
        <section className="report-section">
          <h3><Activity size={17} /> Execution Trace Timeline ({traceEvents.length} Events)</h3>
          <div className="trace-timeline">
            {traceEvents.map((evt, idx) => (
              <div key={idx} className="trace-row">
                <span className={`trace-dot ${evt.status === 'error' ? 'err' : 'ok'}`} />
                <span className="trace-stage">{evt.stage}</span>
                <span className="trace-msg">{evt.message}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function isInternalNote(note: string): boolean {
  const s = note.toLowerCase();
  return [
    'only select statements are permitted',
    'exceeded maximum row limit',
    'add a limit clause',
    'max_refinements',
    'wall-clock budget',
    'without repeating successful work',
    'two bounded aggregate queries',
    'detail requests failed',
    'ai narrative commentary unavailable',
  ].some(p => s.includes(p));
}

function isRawDumpFinding(detail: string): boolean {
  return detail.includes("[{'") || detail.includes("={'") || (detail.match(/\b[a-z_]+=[0-9.-]+/g) || []).length >= 3;
}

function curateFrontendTables(tables: Array<{ title: string; rows: Array<Record<string, unknown>> }>) {
  const candidates = tables
    .filter(t => Array.isArray(t?.rows) && t.rows.length > 0)
    .filter(t => {
      const tl = (t.title || '').toLowerCase();
      if (!['reconciliation', 'audit', 'check'].some(w => tl.includes(w))) return true;
      const diffCols = Object.keys(t.rows[0] || {}).filter(c => ['diff', 'discrepancy', 'residual'].some(w => c.toLowerCase().includes(w)));
      if (!diffCols.length) return true;
      return !t.rows.every(r => diffCols.every(c => typeof r[c] !== 'number' || Math.abs(Number(r[c])) <= 1e-4));
    })
    .map(t => {
      const cols = Object.keys(t.rows[0] || {});
      const hasPeriodLabel = cols.some(c => ['period_label', 'period', 'month', 'label'].includes(c.toLowerCase()));
      const dropCols = new Set(cols.filter(c => hasPeriodLabel && ['period_start', 'period_end', 'period_end_exclusive'].includes(c.toLowerCase())));
      if (!dropCols.size) return t;
      return {
        ...t,
        rows: t.rows.map(r => Object.fromEntries(Object.entries(r).filter(([k]) => !dropCols.has(k)))),
      };
    });

  const firstColSig = (rows: Array<Record<string, unknown>>) => {
    const fc = Object.keys(rows[0] || {})[0];
    return fc ? rows.map(r => String(r[fc] ?? '').toLowerCase()).join('|') : '';
  };

  return candidates.filter((t, idx) => {
    const isGeneric = ['query output', 'gross revenue summary', 'gross merchandise revenue summary'].includes((t.title || '').toLowerCase());
    if (!isGeneric) return true;
    const sig = firstColSig(t.rows);
    return !candidates.some((other, j) => j !== idx && firstColSig(other.rows) === sig);
  });
}

function ReportView({ response, onFollowUp }: { response: ChatResponse; onFollowUp: (q: string) => void }) {
  const report: Report = response.report;
  const validTables = curateFrontendTables(report.tables || []);
  const rawSections = report.business_report?.sections || [];
  const cleanSections = rawSections.filter(
    s => !(rawSections.length >= 3 && /methodology|accounting rules|data scope/i.test(s.heading))
  );
  const cleanFindings = (report.findings || []).filter(f => f.detail && !isRawDumpFinding(f.detail));
  const cleanLimitations = (report.limitations || []).filter(l => l && !isInternalNote(l)).slice(0, 4);
  const provenanceCount = (report.python_analyses?.length || 0) + (report.evidence?.length || 0) + (report.models_used?.length || 0);
  const [activeTab, setActiveTab] = useState<'overview' | 'tables' | 'provenance'>('overview');
  const isDemoRun = response.run_id.startsWith('demo-');

  useEffect(() => setActiveTab('overview'), [response.run_id]);
  const maxDriverEffect = Math.max(1, ...(report.hypotheses || []).map(h => Math.abs(Number(h.measured_effect) || 0)));

  return (
    <div className="report">
      <div className="report-hero">
        <div className="report-topline">
          <div className="report-badges">
            <span className="hero-pill"><Sparkles size={13} /> {isDemoRun ? 'Interactive Demo Showcase' : 'Executive Report'}</span>
            {report.period && (
              <span className="hero-pill"><Clock3 size={13} /> {report.period}{report.comparison ? ` vs. ${report.comparison}` : ''}</span>
            )}
          </div>

          <div className="report-actions">
            {isDemoRun ? (
              <>
                <button type="button" className="export-btn" onClick={() => openDemoReportWindow(response, false)}><FileText size={14} /> View Report</button>
                <button type="button" className="export-btn primary" onClick={() => openDemoReportWindow(response, true)}><Download size={14} /> Print / Save PDF</button>
              </>
            ) : (
              <>
                <a className="export-btn" href={api.reportHtmlUrl(response.run_id)} target="_blank" rel="noreferrer"><FileText size={14} /> View Report</a>
                <a className="export-btn primary" href={api.reportPdfUrl(response.run_id)} target="_blank" rel="noreferrer"><Download size={14} /> Download PDF</a>
              </>
            )}
          </div>
        </div>

        <h2>{stripMarkdown(report.title)}</h2>
      </div>

      <div className="report-body">
        <div className="summary-callout">
          <div className="summary-callout-label"><Sparkles size={14} /> Executive Summary</div>
          <RichText text={report.summary} />
        </div>

        {report.kpis.length > 0 && (
          <div className="kpi-grid">
            {report.kpis.map((k, i) => {
              const isNeg = k.value !== null && k.value < 0;
              const isPosPct = k.unit === '%' && k.value !== null && k.value > 0;
              return (
                <div className="kpi" key={i}>
                  <div className="kpi-label">{k.label}</div>
                  <strong className={isNeg ? 'negative' : isPosPct ? 'positive' : ''}>{displayNumber(k.value, k.unit)}</strong>
                </div>
              );
            })}
          </div>
        )}

        <div className="tabs" role="tablist">
          <button className={activeTab === 'overview' ? 'selected' : ''} onClick={() => setActiveTab('overview')}>
            <BarChart3 size={15} /> Analysis & Charts
          </button>
          {validTables.length > 0 && (
            <button className={activeTab === 'tables' ? 'selected' : ''} onClick={() => setActiveTab('tables')}>
              <Table2 size={15} /> Data Tables <span>{validTables.length}</span>
            </button>
          )}
          <button className={activeTab === 'provenance' ? 'selected' : ''} onClick={() => setActiveTab('provenance')}>
            <ShieldCheck size={15} /> Methodology, Code & Audit Trace {provenanceCount > 0 && <span>{provenanceCount}</span>}
          </button>
        </div>

        {activeTab === 'overview' && (
          <>
            {report.charts.length > 0 && (
              <div className="charts-grid">{report.charts.map((chart, i) => <DataChart key={i} chart={chart} />)}</div>
            )}

            {cleanSections.length > 0 ? (
              <section className="report-section">
                <h3><FileText size={17} /> Analysis & Findings</h3>
                <div className="briefing-grid">
                  {cleanSections.map((sec, idx) => (
                    <div className="briefing-card" key={idx}>
                      <div className="briefing-num">{String(idx + 1).padStart(2, '0')}</div>
                      <div className="briefing-body">
                        <h4>{stripMarkdown(sec.heading).replace(/^\d+[.)]\s*/, '')}</h4>
                        <RichText text={sec.body} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : cleanFindings.length > 0 ? (
              <section className="report-section">
                <h3><Sparkles size={17} /> Key Findings</h3>
                <div className="finding-list">
                  {cleanFindings.map((item, i) => (
                    <div className={`finding ${item.status}`} key={i}>
                      <div className="finding-body">
                        <div className="finding-header">
                          <strong>{stripMarkdown(item.title).replace(/^Autonomous Python Analysis:\s*/i, '')}</strong>
                        </div>
                        <RichText text={item.detail} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : null}

            {report.hypotheses && report.hypotheses.length > 0 && (
              <section className="report-section">
                <h3><TrendingUp size={17} /> Driver & Hypothesis Evaluation</h3>
                <div className="finding-list">
                  {report.hypotheses.map((h, i) => {
                    const isSupported = h.status === 'verified' || h.status === 'supported';
                    const effectVal = Number(h.measured_effect) || 0;
                    const barPct = effectVal !== 0 ? Math.min(100, Math.max(8, Math.round((Math.abs(effectVal) / maxDriverEffect) * 100))) : 18;
                    const barColor = effectVal < 0 ? '#dc2626' : '#059669';

                    return (
                      <div className={`finding ${isSupported ? 'verified' : 'warning'}`} key={i}>
                        <div className={`finding-icon ${isSupported ? 'verified' : 'warning'}`}>
                          {isSupported ? <CheckCheck size={16} /> : <CircleAlert size={16} />}
                        </div>
                        <div className="finding-body">
                          <div className="finding-header">
                            <strong>{stripMarkdown(h.description)}</strong>
                            <span className={`state-pill ${isSupported ? 'good' : 'warn'}`}>{formatLabel(h.status)}</span>
                          </div>
                          <div className="driver-meta">
                            {h.dimension && <span className="driver-tag">Dimension: {formatLabel(h.dimension)}</span>}
                            {effectVal !== 0 && <span className="driver-tag">Measured Impact: <strong>{currencyFormatter.format(effectVal)}</strong></span>}
                          </div>
                          {effectVal !== 0 && (
                            <div className="impact-bar-wrap">
                              <div className="impact-track">
                                <div className="impact-fill" style={{ width: `${barPct}%`, backgroundColor: barColor }} />
                              </div>
                            </div>
                          )}
                          {h.limitations && <RichText text={h.limitations} />}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {validTables.map((table, i) => <FormattedDataTable key={i} title={table.title} rows={table.rows} />)}

            {report.recommendations.length > 0 && (
              <section className="report-section">
                <h3><Lightbulb size={17} /> Recommendations</h3>
                <div className="recommendations-list">
                  {report.recommendations.map((r, i) => (
                    <div className="recommendation-card" key={i}>
                      <span className="recommendation-num">{i + 1}</span>
                      <p>{stripMarkdown(r)}</p>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {report.follow_up_questions && report.follow_up_questions.length > 0 && (
              <section className="followup-section">
                <h4>Explore Further</h4>
                <div className="followup-chips">
                  {report.follow_up_questions.map((fq, idx) => (
                    <button key={idx} className="followup-chip" onClick={() => onFollowUp(fq)}>
                      <ArrowRight size={13} /><span>{stripMarkdown(fq)}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {cleanLimitations.length > 0 && (
              <section className="limitations">
                <h3><CircleAlert size={15} /> Scope & Methodology Notes</h3>
                {cleanLimitations.map((item, i) => <p key={i}>{stripMarkdown(item)}</p>)}
              </section>
            )}
          </>
        )}

        {activeTab === 'tables' && (
          <div>
            {validTables.map((table, i) => (
              <FormattedDataTable key={i} title={table.title} rows={table.rows} style={{ marginTop: i === 0 ? 0 : '20px' }} />
            ))}
          </div>
        )}

        {activeTab === 'provenance' && <ProvenanceView response={response} />}
      </div>
    </div>
  );
}

function App() {
  const [question, setQuestion] = useState('');
  const [session, setSession] = useState<string | null>(localStorage.getItem('analyst_session'));
  const [turns, setTurns] = useState<Turn[]>([]);
  const [current, setCurrent] = useState<ChatResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [liveEvents, setLiveEvents] = useState<TraceEvent[]>([]);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (!session || session === 'demo-session-showcase') return;
    api
      .history(session)
      .then(runs => {
        const restored = runs.filter(r => r.question).map(r => ({ question: r.question!, response: r }));
        setTurns(restored);
        setCurrent(restored.length ? restored[restored.length - 1].response : null);
      })
      .catch(() => {
        localStorage.removeItem('analyst_session');
        setSession(null);
      });
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        newChat();
        input.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  async function runDemoShowcase(showcase: DemoShowcase) {
    if (busy) return;
    setError(null);
    setBusy(true);
    setLiveEvents([]);
    setSidebar(false);
    for (const evt of showcase.response.report.trace || []) {
      await new Promise(resolve => setTimeout(resolve, 175));
      setLiveEvents(prev => [...prev, evt]);
    }
    setTurns(old => [...old.filter(t => t.response.run_id !== showcase.response.run_id), { question: showcase.question, response: showcase.response }]);
    setCurrent(showcase.response);
    setBusy(false);
  }

  async function submit(text = question) {
    const q = text.trim();
    if (!q || busy) return;
    if (!health) {
      const matched = DEMO_SHOWCASES.find(s => s.question.toLowerCase() === q.toLowerCase());
      if (matched) {
        setQuestion('');
        await runDemoShowcase(matched);
        return;
      }
    }
    setError(null);
    setBusy(true);
    setLiveEvents([]);
    setQuestion('');
    try {
      const reply = await api.askStream(q, session, evt => setLiveEvents(prev => [...prev, evt]));
      setSession(reply.session_id);
      localStorage.setItem('analyst_session', reply.session_id);
      setTurns(old => [...old, { question: q, response: reply }]);
      setCurrent(reply);
    } catch (err: unknown) {
      const errorObj = err as Error & { code?: string };
      setError({ code: errorObj.code || 'REQUEST_FAILED', message: errorObj.message || 'We could not complete your request right now. Please try again.' });
      setQuestion(q);
    } finally {
      setBusy(false);
    }
  }

  function newChat() {
    setSession(null);
    localStorage.removeItem('analyst_session');
    setTurns([]);
    setCurrent(null);
    setQuestion('');
    setError(null);
    setLiveEvents([]);
    setSidebar(false);
  }

  const activeStepIdx = resolveProgressStepIndex(liveEvents);
  const activeStepLabel = PROGRESS_STEPS[activeStepIdx] || 'Analyzing your question';

  return (
    <div className="shell">
      <aside className={`sidebar ${sidebar ? 'visible' : ''}`}>
        <div className="logo">
          <div className="logo-icon"><Layers3 size={21} /></div>
          <div><strong>atlas<span>.</span></strong><small>BUSINESS ANALYTICS</small></div>
          <button onClick={() => setSidebar(false)} className="sidebar-close" aria-label="Close sidebar"><X size={18} /></button>
        </div>

        <button className="new-chat" onClick={newChat}><Plus size={17} /> New Analysis <span>⌘K</span></button>

        <div className="side-label">Instant Demo Showcases</div>
        <div className="demo-sidebar-list">
          {DEMO_SHOWCASES.map(sc => (
            <button
              key={sc.id}
              type="button"
              disabled={busy}
              className={`recent-item demo-item ${current?.run_id === sc.response.run_id ? 'active' : ''}`}
              onClick={() => runDemoShowcase(sc)}
            >
              <Play size={14} /><span>{sc.badge}</span>
            </button>
          ))}
        </div>

        <div className="side-label">Recent Analyses</div>
        <div className="recent-list">
          {turns.slice(-8).reverse().map((t, i) => (
            <button
              key={i}
              className={`recent-item ${current?.run_id === t.response.run_id ? 'active' : ''}`}
              onClick={() => { setCurrent(t.response); setSidebar(false); }}
            >
              <MessageSquare size={15} /><span>{t.question}</span>
            </button>
          ))}
          {!turns.length && <p className="empty-recent">Your recent questions and reports will appear here for quick access.</p>}
        </div>

        <div className="sidebar-bottom">
          <div className="source-status">
            <span className={health ? 'green-dot' : 'amber-dot'} />
            <div>
              <strong>Sales & Operations Data</strong>
              <small>{health ? 'Connected · Ready for live queries' : 'Demo Replay Ready · Start backend for live SQL'}</small>
            </div>
          </div>
        </div>
      </aside>

      {sidebar && <button className="screen-mask" onClick={() => setSidebar(false)} aria-label="Close navigation" />}

      <main className="main">
        <header className="topbar">
          <button className="menu" onClick={() => setSidebar(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="crumb">
            <span>Analytics Workspace</span><ChevronRight size={15} />
            <strong>{current ? stripMarkdown(current.report.title) : 'New Analysis'}</strong>
          </div>
          <div className="top-actions">
            <span className="mode-pill">
              <span className={health ? 'green-dot' : 'amber-dot'} />
              {health ? 'Live Multi-Agent Connected' : 'Interactive Demo Mode'}
            </span>
          </div>
        </header>

        <div className="page-scroll">
          <div className="content">
            {busy && (
              <div className="progress-card" role="status" aria-live="polite">
                <div className="progress-header">
                  <span className="spinner-green" />
                  <strong>{activeStepLabel}...</strong>
                  <span className="sub">{liveEvents.length > 0 ? liveEvents[liveEvents.length - 1].message : 'Building your report'}</span>
                </div>
                <div className="progress-steps">
                  {PROGRESS_STEPS.map((label, idx) => (
                    <span key={label} className={`progress-step ${idx < activeStepIdx ? 'done' : idx === activeStepIdx ? 'active' : ''}`}>
                      {idx < activeStepIdx && <Check size={13} />}{label}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {!current ? (
              <div className="welcome">
                <div className="welcome-badge"><Sparkles size={14} /> Hierarchical Multi-Agent AI Data Analyst</div>
                <h1>Turn business data into <em>verified decisions.</em></h1>
                <p>
                  Ask questions in plain English about revenue, product performance, customer cohorts, or period comparisons
                  to get instant charts, SHA-256 reproduced sandbox code, and executive reports.
                </p>

                <div className="showcase-banner">
                  <div className="showcase-banner-head">
                    <span><Play size={14} /> Instant Interactive Demo Showcases (Zero Setup Required)</span>
                    <small>Replay verified multi-agent runs with SQL, Python code & Plotly charts</small>
                  </div>
                  <div className="showcase-cards">
                    {DEMO_SHOWCASES.map(sc => (
                      <button key={sc.id} type="button" disabled={busy} className="showcase-card" onClick={() => runDemoShowcase(sc)}>
                        <div className="showcase-card-top">
                          <span className="showcase-pill">{sc.badge}</span>
                          <span className="showcase-play"><Play size={12} /> Replay Run</span>
                        </div>
                        <strong>{sc.title}</strong>
                        <small>{sc.description}</small>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="prompt-grid">
                  {PROMPTS.map((item, i) => (
                    <button key={i} className="prompt-card" onClick={() => submit(item.text)} disabled={busy}>
                      <div className="prompt-icon"><item.icon size={19} /></div>
                      <div className="prompt-card-body">
                        <span className="prompt-tag">{item.tag}</span>
                        <strong>{item.text}</strong>
                        <small>{item.note}</small>
                      </div>
                      <ArrowRight size={17} className="prompt-arrow" />
                    </button>
                  ))}
                </div>

                <div className="feature-row">
                  <span><BarChart3 size={15} /> Interactive Visuals</span>
                  <span><ShieldCheck size={15} /> 3-Layer SHA-256 Sandbox Reproduction</span>
                  <span><FileText size={15} /> Executive PDF & HTML Reports</span>
                </div>
              </div>
            ) : (
              <div className="conversation">
                <div className="question-bar">
                  <div className="question-bar-left">
                    <div className="user-avatar">Q</div>
                    <div className="question-text">{turns.find(t => t.response.run_id === current.run_id)?.question || 'Analysis Request'}</div>
                  </div>
                </div>
                <ReportView response={current} onFollowUp={q => submit(q)} />
              </div>
            )}

            {error && (
              <div className="alert" role="alert">
                <div className="alert-left">
                  <CircleAlert size={18} />
                  <div><strong>Unable to complete live analysis</strong><p>{error.message}</p></div>
                </div>
                <div className="alert-actions">
                  <button type="button" onClick={() => runDemoShowcase(DEMO_SHOWCASES[0])} className="retry-btn demo-fallback-btn">
                    <Play size={13} /> Launch Demo Showcase
                  </button>
                  <button onClick={() => submit(question)} className="retry-btn"><RefreshCw size={13} /> Try again</button>
                </div>
              </div>
            )}

            <div className="composer-wrap">
              <div className="composer">
                <textarea
                  rows={1}
                  ref={input}
                  value={question}
                  disabled={busy}
                  placeholder="Ask a question about sales, product performance, or revenue trends..."
                  onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      submit();
                    }
                  }}
                />
                <button className="composer-send" disabled={busy || !question.trim()} onClick={() => submit()} aria-label="Submit question">
                  {busy ? <span className="spinner" /> : <Send size={17} />}
                </button>
              </div>
              <div className="composer-foot">
                <span>Ask any question about your revenue, products, orders, or customers</span>
                <span>{busy ? `${activeStepLabel}...` : 'Press Enter to ask · Shift+Enter for new line'}</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

import { useEffect, useRef, useState } from 'react';
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
import type { Chart, ChatResponse, Report, TraceEvent, Turn } from './types';

const PROMPTS = [
  { icon: TrendingDown, tag: 'Root Cause Analysis', text: 'Why did sales decline in November 2004 compared with October 2004?', note: 'Identify key drivers across product lines, regions, and order volume' },
  { icon: BarChart3, tag: 'Top Performers', text: 'Show the top 10 products by revenue in 2004', note: 'Rank best-selling products with revenue breakdown and visual comparison' },
  { icon: TrendingUp, tag: 'Year-over-Year Comparison', text: 'Compare 2004 vs 2003 revenue by product line', note: 'Analyze annual revenue growth, category shifts, and performance trends' },
  { icon: FileText, tag: 'Executive Summary', text: 'Generate a comprehensive executive business report on November 2004 revenue performance', note: 'Full executive overview with key metrics, findings, and PDF export' },
];

const PROGRESS_STEPS = ['Understanding question', 'Gathering business data', 'Analyzing trends & drivers', 'Preparing charts & insights'];
const usdFmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
const numFmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

function displayNumber(value: number | null, unit = '') {
  if (value == null) return '—';
  if (unit === 'USD') return usdFmt.format(value);
  if (unit === '%') return `${value > 0 ? '+' : ''}${numFmt.format(value)}%`;
  return unit ? `${numFmt.format(value)} ${unit}` : numFmt.format(value);
}

function formatLabel(raw = ''): string {
  return raw.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function stripMd(raw = ''): string {
  return raw.replace(/\*\*(.+?)\*\*/g, '$1').replace(/`(.+?)`/g, '$1').replace(/^#{1,6}\s+/gm, '').trim();
}

function RichText({ text }: { text?: string }) {
  if (!text) return null;
  return (
    <div className="rich-text">
      {text.split(/\r?\n/).filter(Boolean).map((raw, i) => {
        const line = raw.trim().replace(/^#{1,6}\s+|^[-*•]\s+/, '');
        const parts = line.split(/(\*\*[^*]+\*\*)/g).map((p, j) =>
          p.startsWith('**') && p.endsWith('**') ? <strong key={j}>{p.slice(2, -2)}</strong> : p.replace(/`([^`]+)`/g, '$1')
        );
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}

function resolveProgressStepIndex(events: TraceEvent[]): number {
  const latest = (events[events.length - 1]?.stage || '').toLowerCase();
  if (['insight', 'synthesize', 'report', 'chart', 'quality'].some(k => latest.includes(k))) return 3;
  if (['python', 'hypothesis', 'review', 'aggregate', 'refinement'].some(k => latest.includes(k))) return 2;
  if (['sql', 'exec', 'supervisor'].some(k => latest.includes(k))) return 1;
  return 0;
}

function DataChart({ chart }: { chart: Chart }) {
  const [mode, setMode] = useState<'bar' | 'line'>(chart.kind === 'line' ? 'line' : 'bar');
  useEffect(() => setMode(chart.kind === 'line' ? 'line' : 'bar'), [chart.kind, chart.title]);

  const isLine = mode === 'line';
  const labels = chart.labels || [];
  const values = (chart.values || []).map(v => Number(v) || 0);
  const height = isLine ? 310 : Math.max(280, Math.min(440, labels.length * 32 + 90));

  const data = isLine
    ? {
        type: 'scatter' as const, mode: 'lines+markers' as const, x: labels, y: values,
        line: { color: '#059669', width: 3, shape: 'spline' as const }, marker: { color: '#059669', size: 7 },
        fill: 'tozeroy' as const, fillcolor: 'rgba(16, 185, 129, 0.11)',
      }
    : {
        type: 'bar' as const, orientation: 'h' as const, y: labels, x: values,
        text: values.map(v => displayNumber(v, chart.unit)), textposition: 'auto' as const,
        marker: { color: values.map(v => (v < 0 ? '#dc2626' : '#059669')) },
      };

  return (
    <div className="chart-panel">
      <div className="panel-heading">
        <div className="panel-heading-title">{chart.title}</div>
        <div className="chart-type-toggle">
          <button type="button" className={!isLine ? 'active' : ''} onClick={() => setMode('bar')}>Bar</button>
          <button type="button" className={isLine ? 'active' : ''} onClick={() => setMode('line')}>Trend</button>
        </div>
      </div>
      <Plot
        data={[data]}
        layout={{
          autosize: true, height, margin: { l: isLine ? 58 : 175, r: 28, t: 12, b: 48 },
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
          font: { family: 'Inter, system-ui, sans-serif', color: '#475569', size: 11.5 },
          xaxis: { gridcolor: '#f1f5f3', zerolinecolor: '#94a3b8', automargin: true },
          yaxis: { gridcolor: isLine ? '#f1f5f3' : 'transparent', automargin: true, autorange: isLine ? undefined : 'reversed' },
        }}
        style={{ width: '100%', height }}
        useResizeHandler
        config={{ displayModeBar: false, responsive: true }}
      />
    </div>
  );
}

function downloadTableCsv(title: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc = (v: unknown) => {
    const s = String(v ?? '');
    return /[,"\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const csv = [cols.map(esc).join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
  Object.assign(document.createElement('a'), { href: url, download: `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.csv` }).click();
  URL.revokeObjectURL(url);
}

function formatCell(val: unknown, colKey: string): string {
  if (val == null || val === '') return '—';
  if (typeof val === 'string') return val.includes('_') ? formatLabel(val) : val;
  if (typeof val !== 'number') return String(val);
  const lk = colKey.toLowerCase();
  const sign = ['change', 'variance', 'delta'].some(t => lk.includes(t)) && val > 0 ? '+' : '';
  if (['pct', 'percent', 'share', 'rate'].some(t => lk.includes(t))) return `${sign}${numFmt.format(val)}%`;
  if (['revenue', 'amount', 'price', 'value', 'previous', 'current', 'baseline', 'change', 'variance', 'aov', 'monetary', 'usd'].some(t => lk.includes(t))) {
    return `${sign}${usdFmt.format(val)}`;
  }
  return `${sign}${numFmt.format(val)}`;
}

function FormattedDataTable({ title, rows }: { title: string; rows: Record<string, unknown>[] }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  const isNumCol = (c: string) => rows.some(r => typeof r[c] === 'number');

  return (
    <section className="data-table">
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
            <tr>{cols.map(k => <th key={k} className={isNumCol(k) ? 'num' : ''}>{formatLabel(k)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, j) => (
              <tr key={j}>
                {cols.map(key => {
                  const val = row[key];
                  const isNum = typeof val === 'number';
                  const isVar = isNum && ['change', 'variance', 'pct', 'delta'].some(t => key.toLowerCase().includes(t));
                  const tone = isVar ? ((val as number) < 0 ? 'neg-val' : (val as number) > 0 ? 'pos-val' : '') : '';
                  return <td key={key} className={`${isNum ? 'num' : ''} ${tone}`.trim()}>{formatCell(val, key)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="copy-code-btn"
      onClick={() => {
        navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? 'Copied' : label}
    </button>
  );
}

function ProvenanceView({ response }: { response: ChatResponse }) {
  const { report } = response;
  const pythonAnalyses = report.python_analyses || [];
  const evidenceList = report.evidence || [];
  const modelsUsed = report.models_used || [];
  const taskGraph = report.task_graph || [];
  const traceEvents = report.trace || [];
  const totalTokens = modelsUsed.reduce((acc, m) => acc + (Number(m.tokens) || 0), 0);
  const totalCost = modelsUsed.reduce((acc, m) => acc + (Number(m.cost_usd) || 0), 0);

  return (
    <div className="provenance-stack">
      <section className="report-section provenance-banner">
        <div className="provenance-banner-top">
          <div className="provenance-title-group">
            <ShieldCheck size={20} />
            <div>
              <h4>3-Layer Independent Verification & Guardrail Audit</h4>
              <p>SQL queries are AST-validated via SQLGlot, Python runs in an isolated child process, and numerical claims are verified against SHA-256 dataset snapshots.</p>
            </div>
          </div>
          <span className="state-pill good"><CheckCheck size={13} /> {formatLabel(report.evidence_review?.decision || report.status)}</span>
        </div>
        <div className="provenance-metrics-row">
          <div className="prov-metric"><span>Jev Route</span><strong>{response.route}</strong><small>{Math.round((response.route_confidence || 1) * 100)}% confidence</small></div>
          <div className="prov-metric"><span>SQL Snapshots</span><strong>{evidenceList.length} Queries</strong><small>Read-only AST verified</small></div>
          <div className="prov-metric"><span>Python Sandbox</span><strong>{pythonAnalyses.length} Executed</strong><small>Isolated Process Sandbox</small></div>
          <div className="prov-metric"><span>Token Usage</span><strong>{totalTokens.toLocaleString()} tokens</strong><small>{totalCost > 0 ? `$${totalCost.toFixed(4)} est. cost` : `${modelsUsed.length} calls`}</small></div>
        </div>
      </section>

      {taskGraph.length > 0 && (
        <section className="report-section">
          <h3><GitBranch size={17} /> Task DAG ({taskGraph.length} Tasks)</h3>
          <div className="dag-grid">
            {taskGraph.map(t => (
              <div key={t.id} className="dag-card">
                <div className="dag-card-header">
                  <span className={`agent-badge agent-${t.agent}`}>{t.agent}</span>
                  <span className={`state-pill ${t.status === 'COMPLETED' ? 'good' : 'warn'}`}>{t.status}</span>
                </div>
                <strong className="dag-task-title">{t.title}</strong>
                <div className="dag-task-meta"><code>{t.id}</code><span>{t.duration_ms} ms</span></div>
              </div>
            ))}
          </div>
        </section>
      )}

      {pythonAnalyses.length > 0 && (
        <section className="report-section">
          <h3><Terminal size={17} /> Python Sandbox Execution ({pythonAnalyses.length})</h3>
          <div className="prov-card-list">
            {pythonAnalyses.map((py, idx) => (
              <div key={idx} className="prov-code-card">
                <div className="prov-code-header">
                  <div>
                    <div className="prov-code-badges">
                      <span className="agent-badge agent-python"><Code2 size={12} /> {py.task_id}</span>
                      <span className="meta-chip">{py.duration_ms} ms</span>
                    </div>
                    <h4 className="prov-method-title">{py.methodology || formatLabel(py.tool)}</h4>
                  </div>
                  {py.generated_code && <CopyButton text={py.generated_code} label="Copy Python" />}
                </div>
                {py.generated_code && <pre className="code-block"><code>{py.generated_code}</code></pre>}
              </div>
            ))}
          </div>
        </section>
      )}

      {evidenceList.length > 0 && (
        <section className="report-section">
          <h3><Database size={17} /> Executed Read-Only SQL ({evidenceList.length})</h3>
          <div className="prov-card-list">
            {evidenceList.map((ev, idx) => (
              <div key={idx} className="prov-code-card">
                <div className="prov-code-header">
                  <div>
                    <div className="prov-code-badges">
                      <span className="agent-badge agent-sql">{ev.task_id}</span>
                      <span className="meta-chip">{ev.rows?.length || 0} rows · {ev.duration_ms} ms</span>
                    </div>
                    <h4 className="prov-method-title">{ev.purpose || ev.query_id}</h4>
                  </div>
                  {ev.sql && <CopyButton text={ev.sql} label="Copy SQL" />}
                </div>
                {ev.sql && <pre className="code-block"><code>{ev.sql}</code></pre>}
              </div>
            ))}
          </div>
        </section>
      )}

      {modelsUsed.length > 0 && (
        <section className="data-table">
          <div className="data-table-header"><h3><Cpu size={16} /> Model Routing & Telemetry</h3></div>
          <div className="table-scroller">
            <table>
              <thead><tr><th>Agent Role</th><th>Task</th><th>Model</th><th>Effort</th><th className="num">Latency</th><th className="num">Tokens</th></tr></thead>
              <tbody>
                {modelsUsed.map((m, idx) => (
                  <tr key={idx}>
                    <td><strong>{m.agent_role}</strong></td><td>{formatLabel(m.task_type)}</td><td><code>{m.model}</code></td>
                    <td>{formatLabel(m.reasoning_effort)}</td><td className="num">{m.duration_ms} ms</td><td className="num">{(m.tokens || 0).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {traceEvents.length > 0 && (
        <section className="report-section">
          <h3><Activity size={17} /> Execution Trace ({traceEvents.length})</h3>
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

function ReportView({ response, onFollowUp }: { response: ChatResponse; onFollowUp: (q: string) => void }) {
  const report: Report = response.report;
  const tables = (report.tables || []).filter(t => Array.isArray(t?.rows) && t.rows.length > 0);
  const sections = report.business_report?.sections || [];
  const findings = report.findings || [];
  const limitations = (report.limitations || []).slice(0, 4);
  const provenanceCount = (report.python_analyses?.length || 0) + (report.evidence?.length || 0) + (report.models_used?.length || 0);
  const [activeTab, setActiveTab] = useState<'overview' | 'tables' | 'provenance'>('overview');
  const isDemo = response.run_id.startsWith('demo-');

  useEffect(() => setActiveTab('overview'), [response.run_id]);
  const maxEffect = Math.max(1, ...(report.hypotheses || []).map(h => Math.abs(Number(h.measured_effect) || 0)));

  return (
    <div className="report">
      <div className="report-hero">
        <div className="report-topline">
          <div className="report-badges">
            <span className="hero-pill"><Sparkles size={13} /> {isDemo ? 'Interactive Demo Showcase' : 'Executive Report'}</span>
            {report.period && <span className="hero-pill"><Clock3 size={13} /> {report.period}{report.comparison ? ` vs. ${report.comparison}` : ''}</span>}
          </div>
          <div className="report-actions">
            {isDemo ? (
              <button type="button" className="export-btn primary" onClick={() => window.print()}><Download size={14} /> Print / Save PDF</button>
            ) : (
              <>
                <a className="export-btn" href={api.reportHtmlUrl(response.run_id)} target="_blank" rel="noreferrer"><FileText size={14} /> View Report</a>
                <a className="export-btn primary" href={api.reportPdfUrl(response.run_id)} target="_blank" rel="noreferrer"><Download size={14} /> Download PDF</a>
              </>
            )}
          </div>
        </div>
        <h2>{stripMd(report.title)}</h2>
      </div>

      <div className="report-body">
        <div className="summary-callout">
          <div className="summary-callout-label"><Sparkles size={14} /> Executive Summary</div>
          <RichText text={report.summary} />
        </div>

        {report.kpis.length > 0 && (
          <div className="kpi-grid">
            {report.kpis.map((k, i) => (
              <div className="kpi" key={i}>
                <div className="kpi-label">{k.label}</div>
                <strong className={k.value !== null && k.value < 0 ? 'negative' : k.unit === '%' && k.value !== null && k.value > 0 ? 'positive' : ''}>
                  {displayNumber(k.value, k.unit)}
                </strong>
              </div>
            ))}
          </div>
        )}

        <div className="tabs" role="tablist">
          <button className={activeTab === 'overview' ? 'selected' : ''} onClick={() => setActiveTab('overview')}>
            <BarChart3 size={15} /> Analysis & Charts
          </button>
          {tables.length > 0 && (
            <button className={activeTab === 'tables' ? 'selected' : ''} onClick={() => setActiveTab('tables')}>
              <Table2 size={15} /> Data Tables <span>{tables.length}</span>
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

            {sections.length > 0 ? (
              <section className="report-section">
                <h3><FileText size={17} /> Analysis & Findings</h3>
                <div className="briefing-grid">
                  {sections.map((sec, idx) => (
                    <div className="briefing-card" key={idx}>
                      <div className="briefing-num">{String(idx + 1).padStart(2, '0')}</div>
                      <div className="briefing-body">
                        <h4>{stripMd(sec.heading)}</h4>
                        <RichText text={sec.body} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : findings.length > 0 ? (
              <section className="report-section">
                <h3><Sparkles size={17} /> Key Findings</h3>
                <div className="finding-list">
                  {findings.map((item, i) => (
                    <div className={`finding ${item.status}`} key={i}>
                      <div className="finding-body">
                        <strong>{stripMd(item.title)}</strong>
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
                    const ok = h.status === 'verified' || h.status === 'supported';
                    const eff = Number(h.measured_effect) || 0;
                    const pct = eff !== 0 ? Math.min(100, Math.max(8, Math.round((Math.abs(eff) / maxEffect) * 100))) : 18;
                    return (
                      <div className={`finding ${ok ? 'verified' : 'warning'}`} key={i}>
                        <div className={`finding-icon ${ok ? 'verified' : 'warning'}`}>{ok ? <CheckCheck size={16} /> : <CircleAlert size={16} />}</div>
                        <div className="finding-body">
                          <div className="finding-header">
                            <strong>{stripMd(h.description)}</strong>
                            <span className={`state-pill ${ok ? 'good' : 'warn'}`}>{formatLabel(h.status)}</span>
                          </div>
                          <div className="driver-meta">
                            {h.dimension && <span className="driver-tag">Dimension: {formatLabel(h.dimension)}</span>}
                            {eff !== 0 && <span className="driver-tag">Impact: <strong>{usdFmt.format(eff)}</strong></span>}
                          </div>
                          {eff !== 0 && (
                            <div className="impact-bar-wrap">
                              <div className="impact-track"><div className="impact-fill" style={{ width: `${pct}%`, backgroundColor: eff < 0 ? '#dc2626' : '#059669' }} /></div>
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

            {tables.map((t, i) => <FormattedDataTable key={i} title={t.title} rows={t.rows} />)}

            {report.recommendations.length > 0 && (
              <section className="report-section">
                <h3><Lightbulb size={17} /> Recommendations</h3>
                <div className="recommendations-list">
                  {report.recommendations.map((r, i) => (
                    <div className="recommendation-card" key={i}>
                      <span className="recommendation-num">{i + 1}</span>
                      <p>{stripMd(r)}</p>
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
                      <ArrowRight size={13} /><span>{stripMd(fq)}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {limitations.length > 0 && (
              <section className="limitations">
                <h3><CircleAlert size={15} /> Scope & Methodology Notes</h3>
                {limitations.map((item, i) => <p key={i}>{stripMd(item)}</p>)}
              </section>
            )}
          </>
        )}

        {activeTab === 'tables' && tables.map((t, i) => <FormattedDataTable key={i} title={t.title} rows={t.rows} />)}
        {activeTab === 'provenance' && <ProvenanceView response={response} />}
      </div>
    </div>
  );
}

export default function App() {
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
  const demoTokenRef = useRef(0);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (!session || session === 'demo-session-showcase') return;
    api.history(session)
      .then(runs => {
        const restored = runs.filter(r => r.question).map(r => ({ question: r.question!, response: r }));
        setTurns(restored);
        setCurrent(restored[restored.length - 1]?.response || null);
      })
      .catch(() => {
        localStorage.removeItem('analyst_session');
        setSession(null);
      });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        newChat();
        input.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  async function runDemoShowcase(showcase: DemoShowcase) {
    if (busy) return;
    const token = ++demoTokenRef.current;
    setError(null);
    setBusy(true);
    setLiveEvents([]);
    setSidebar(false);
    for (const evt of showcase.response.report.trace || []) {
      await new Promise(r => setTimeout(r, 150));
      if (demoTokenRef.current !== token) return;
      setLiveEvents(prev => [...prev, evt]);
    }
    if (demoTokenRef.current !== token) return;
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
        return runDemoShowcase(matched);
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
      const e = err as Error & { code?: string };
      setError({ code: e.code || 'REQUEST_FAILED', message: e.message || 'Request failed.' });
      setQuestion(q);
    } finally {
      setBusy(false);
    }
  }

  function newChat() {
    demoTokenRef.current++;
    setBusy(false);
    setSession(null);
    localStorage.removeItem('analyst_session');
    setTurns([]);
    setCurrent(null);
    setQuestion('');
    setError(null);
    setLiveEvents([]);
    setSidebar(false);
  }

  const stepIdx = resolveProgressStepIndex(liveEvents);
  const stepLabel = PROGRESS_STEPS[stepIdx] || 'Analyzing your question';

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
          {!turns.length && <p className="empty-recent">Your recent questions and reports will appear here.</p>}
        </div>

        <div className="sidebar-bottom">
          <div className="source-status">
            <span className={health ? 'green-dot' : 'amber-dot'} />
            <div>
              <strong>Sales & Operations Data</strong>
              <small>{health ? 'Connected · Ready for live queries' : 'Demo Replay Ready'}</small>
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
            <strong>{current ? stripMd(current.report.title) : 'New Analysis'}</strong>
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
                  <strong>{stepLabel}...</strong>
                  <span className="sub">{liveEvents[liveEvents.length - 1]?.message || 'Building your report'}</span>
                </div>
                <div className="progress-steps">
                  {PROGRESS_STEPS.map((label, idx) => (
                    <span key={label} className={`progress-step ${idx < stepIdx ? 'done' : idx === stepIdx ? 'active' : ''}`}>
                      {idx < stepIdx && <Check size={13} />}{label}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {!current ? (
              <div className="welcome">
                <div className="welcome-badge"><Sparkles size={14} /> Hierarchical Multi-Agent AI Data Analyst</div>
                <h1>Turn business data into <em>verified decisions.</em></h1>
                <p>Ask questions in plain English about revenue, product performance, customer cohorts, or period comparisons.</p>

                <div className="showcase-banner">
                  <div className="showcase-banner-head">
                    <span><Play size={14} /> Instant Interactive Demo Showcases</span>
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
                <span>{busy ? `${stepLabel}...` : 'Press Enter to ask · Shift+Enter for new line'}</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

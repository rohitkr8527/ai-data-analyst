export type Chart = {
  title: string;
  kind: 'bar' | 'line' | 'waterfall' | 'pie' | 'grouped_bar';
  labels: string[];
  values: number[];
  unit: string;
  x_label: string;
  y_label: string;
  source_task_id?: string;
};

export type Finding = {
  title: string;
  detail: string;
  status: 'verified' | 'inconclusive' | 'warning';
  evidence_ids: string[];
};

export type Hypothesis = {
  id: string;
  description: string;
  dimension: string;
  hypothesis_type?: string;
  status: 'verified' | 'inconclusive' | 'warning' | 'supported' | 'not_supported' | 'rejected' | 'revised';
  measured_effect: number;
  share_of_change?: number | null;
  confidence: number;
  evidence_query_ids: string[];
  alternative_explanations?: string[];
  limitations: string;
};

export type Evidence = {
  task_id: string;
  query_id: string;
  purpose: string;
  columns: string[];
  rows: Record<string, unknown>[];
  sql: string;
  duration_ms: number;
  dataset_id?: string | null;
};

export type ModelUsage = {
  agent_role: string;
  task_type: string;
  model: string;
  reasoning_effort: string;
  duration_ms: number;
  tokens: number;
  cost_usd?: number;
};

export type TaskGraphItem = {
  id: string;
  title: string;
  agent: string;
  status: string;
  depends_on: string[];
  priority: number;
  origin: string;
  round: number;
  duration_ms: number;
  error?: string | null;
};

export type PythonExecutionStep = {
  step_index?: number;
  phase?: string;
  action?: string;
  thought?: string;
  status?: string;
  duration_ms?: number;
  observation_summary?: string;
};

export type PythonAnalysisItem = {
  task_id: string;
  tool: string;
  methodology?: string;
  generated_code?: string;
  execution_steps?: PythonExecutionStep[];
  assumptions_checked?: string[];
  repair_attempts?: number;
  verification_status?: string;
  calculation: {
    tool?: string;
    methodology?: string;
    narrative?: string;
    status?: string;
    summary?: Record<string, unknown>;
    records?: Record<string, unknown>[];
    tables?: { title: string; rows: Record<string, unknown>[] }[];
    charts?: Chart[];
    forecasts?: Record<string, unknown>[];
    anomalies?: Record<string, unknown>[];
  };
  duration_ms: number;
};

export type BusinessReportSection = {
  heading: string;
  body: string;
  evidence_ids: string[];
};

export type BusinessReportData = {
  title: string;
  executive_summary: string;
  business_objective: string;
  kpi_overview: string;
  sections: BusinessReportSection[];
  recommendations: string[];
  limitations: string[];
};

export type TraceEvent = {
  stage: string;
  message: string;
  status: string;
  timestamp?: number;
  task_id?: string;
};

export type Report = {
  title: string;
  summary: string;
  status: string;
  output_mode?: string;
  period: string;
  comparison: string;
  kpis: { label: string; value: number | null; unit: string }[];
  findings: Finding[];
  hypotheses?: Hypothesis[];
  charts: Chart[];
  tables: { title: string; rows: Record<string, unknown>[] }[];
  limitations: string[];
  recommendations: string[];
  follow_up_questions?: string[];
  evidence: Evidence[];
  python_analyses?: PythonAnalysisItem[];
  evidence_review?: {
    round?: number;
    sufficient?: boolean;
    decision?: string;
    notes?: string[];
  };
  task_graph?: TaskGraphItem[];
  business_report?: BusinessReportData | null;
  models_used?: ModelUsage[];
  trace: TraceEvent[];
};

export type ChatResponse = {
  run_id: string;
  session_id: string;
  route: string;
  route_confidence: number;
  route_probabilities: Record<string, number>;
  report: Report;
};

export type Turn = {
  question: string;
  response: ChatResponse;
};

import type { ChatResponse } from './types';

export type DemoShowcase = {
  id: string;
  badge: string;
  title: string;
  question: string;
  description: string;
  response: ChatResponse;
};

export const DEMO_SHOWCASES: DemoShowcase[] = [
  {
    id: 'demo-nov-2004-rca',
    badge: 'Multi-Agent Root-Cause DAG',
    title: 'November 2004 vs. October 2004 Sales Variance & Driver Decomposition',
    question: 'Why did sales decline in November 2004 compared with October 2004?',
    description: 'Demonstrates Jev routing, 6-node parallel SQL + Python + Hypothesis DAG, accounting bridge reconciliation, and SHA-256 sandbox reproduction.',
    response: {
      run_id: 'demo-rca-2004-11',
      session_id: 'demo-session-showcase',
      route: 'ROOT_CAUSE_INVESTIGATION',
      route_confidence: 0.97,
      route_probabilities: { ROOT_CAUSE_INVESTIGATION: 0.97, PERIOD_COMPARISON: 0.02, SIMPLE_SQL: 0.01 },
      report: {
        title: 'ClassicModels Gross Merchandise Revenue Root-Cause Analysis (Nov 2004 vs. Oct 2004)',
        summary: 'ClassicModels gross merchandise revenue **increased by +$398,718.67 (+72.07%)** from **$553,257.50** in October 2004 (13 orders) to **$951,976.17** in November 2004 (33 orders). Although the question premised a November decline, baseline reconciliation confirmed November 2004 was the peak Q4 revenue month, while isolated declines occurred in **Ships (-$15,519.55)** and specific European accounts (**Norway -$41,623.44**). The net surge was driven by **+153.85% order volume expansion** (+20 orders), led by **Classic Cars (+$156,743.11)**, **Vintage Cars (+$104,493.36)**, and **USA (+$195,942.73)**.',
        status: 'complete',
        output_mode: 'business_report',
        period: 'November 2004',
        comparison: 'October 2004',
        kpis: [
          { label: 'November 2004 Revenue', value: 951976.17, unit: 'USD' },
          { label: 'October 2004 Revenue', value: 553257.5, unit: 'USD' },
          { label: 'Net Revenue Variance', value: 398718.67, unit: 'USD' },
          { label: 'MoM Revenue Growth', value: 72.07, unit: '%' },
          { label: 'November Order Volume', value: 33, unit: 'orders' },
          { label: 'Average Order Value (Nov)', value: 28847.76, unit: 'USD' },
        ],
        findings: [
          { title: 'Premise Refutation & Baseline Accounting Reconciliation', detail: 'Gross merchandise revenue expanded from **$553,257.50** in October 2004 to **$951,976.17** in November 2004 (**+$398,718.67 / +72.07%**). Exhaustive dimensional bridges across all 7 product lines and 17 active countries reconciled to **$0.00 discrepancy** against the baseline ledger.', status: 'verified', evidence_ids: ['q_baseline_nov_oct', 'ev_py_rev_decomp'] },
          { title: 'Order Volume Expansion Outweighed Average Order Value Compression', detail: 'Logarithmic LMDI decomposition in the Python sandbox confirmed that order count surged from **13 to 33 orders (+153.85%)**, contributing **+$679,492.41** in LMDI volume effect, partially offset by a **-32.21% normalization in average order size** (**-$280,773.74** AOV mix effect) from **$42,558.27** to **$28,847.76**.', status: 'verified', evidence_ids: ['q_baseline_nov_oct', 'ev_py_rev_decomp'] },
          { title: 'Category Pocket of Decline: Ships (-$15,519.55)', detail: 'While 6 of 7 product lines grew—led by **Classic Cars (+$156,743.11)**, **Vintage Cars (+$104,493.36)**, and **Motorcycles (+$68,914.22)**—**Ships** contracted by **-$15,519.55 (-37.19%)**, falling from **$41,732.42** to **$26,212.87**.', status: 'verified', evidence_ids: ['q_product_line_bridge'] },
          { title: 'Geographic Pocket of Decline: European Timing Shifts vs. North American Surge', detail: '**USA (+$195,942.73)** and **France (+$112,418.40)** drove 77.3% of the monthly gain, whereas **Norway (-$41,623.44)**, **Sweden (-$32,110.18)**, and **Germany (-$28,450.10)** experienced month-over-month declines following large October restocking orders.', status: 'verified', evidence_ids: ['q_country_bridge'] },
        ],
        hypotheses: [
          { id: 'hyp_order_volume', description: 'H1: Order transaction volume change (13 vs. 33 orders) is the primary statistical driver of the monthly revenue shift rather than AOV or cancellation spikes.', dimension: 'order_volume', hypothesis_type: 'volume_vs_rate_decomposition', status: 'supported', measured_effect: 398718.67, share_of_change: 100.0, confidence: 0.96, evidence_query_ids: ['q_baseline_nov_oct'], limitations: 'Observational accounting decomposition; correlation does not imply exogenous causation.' },
          { id: 'hyp_ships_contraction', description: 'H2: Isolated contraction in the Ships product line (-$15,519.55) and Nordic accounts (-$73,733.62 combined) created localized sales drops despite overall Q4 expansion.', dimension: 'product_line', hypothesis_type: 'accounting_contribution', status: 'supported', measured_effect: -15519.55, share_of_change: -3.89, confidence: 0.94, evidence_query_ids: ['q_product_line_bridge', 'q_country_bridge'], limitations: 'Product line and country dimensions overlap on European maritime model orders.' },
          { id: 'hyp_cancellation_spike', description: 'H3: Order cancellations or on-hold disputes in November 2004 caused an overall revenue decline.', dimension: 'order_status', hypothesis_type: 'status_anomaly', status: 'rejected', measured_effect: 0.0, share_of_change: 0.0, confidence: 0.99, evidence_query_ids: ['q_baseline_nov_oct'], limitations: 'Rejected by empirical ledger check: 31 of 33 November 2004 orders shipped cleanly with 0 cancelled orders.' },
        ],
        charts: [
          { title: 'Product Line Revenue Variance (Nov 2004 vs. Oct 2004)', kind: 'bar', labels: ['Classic Cars', 'Vintage Cars', 'Motorcycles', 'Planes', 'Trucks and Buses', 'Trains', 'Ships'], values: [156743.11, 104493.36, 68914.22, 46812.4, 31458.95, 5816.18, -15519.55], unit: 'USD', x_label: 'Net Revenue Change (USD)', y_label: 'Product Line', source_task_id: 'product_line_bridge' },
          { title: '2004 Monthly Gross Merchandise Revenue Trajectory', kind: 'line', labels: ['2004-01', '2004-02', '2004-03', '2004-04', '2004-05', '2004-06', '2004-07', '2004-08', '2004-09', '2004-10', '2004-11', '2004-12'], values: [292385.21, 289502.84, 217691.26, 187575.77, 248325.3, 343371.0, 325563.49, 419327.09, 283799.8, 553257.5, 951976.17, 428582.46], unit: 'USD', x_label: 'Month', y_label: 'Gross Revenue (USD)', source_task_id: 'monthly_trend_2004' },
        ],
        tables: [
          {
            title: 'Product Line Variance & Contribution Bridge (Oct 2004 vs. Nov 2004)',
            rows: [
              { product_line: 'Classic Cars', previous: 211450.25, current: 368193.36, change: 156743.11, pct_change: 74.13, share_of_net_change_pct: 39.31 },
              { product_line: 'Vintage Cars', previous: 118920.1, current: 223413.46, change: 104493.36, pct_change: 87.87, share_of_net_change_pct: 26.21 },
              { product_line: 'Motorcycles', previous: 52180.4, current: 121094.62, change: 68914.22, pct_change: 132.07, share_of_net_change_pct: 17.28 },
              { product_line: 'Planes', previous: 58310.15, current: 105122.55, change: 46812.4, pct_change: 80.28, share_of_net_change_pct: 11.74 },
              { product_line: 'Trucks and Buses', previous: 61820.18, current: 93279.13, change: 31458.95, pct_change: 50.89, share_of_net_change_pct: 7.89 },
              { product_line: 'Trains', previous: 8844.0, current: 14660.18, change: 5816.18, pct_change: 65.76, share_of_net_change_pct: 1.46 },
              { product_line: 'Ships', previous: 41732.42, current: 26212.87, change: -15519.55, pct_change: -37.19, share_of_net_change_pct: -3.89 },
            ],
          },
          {
            title: 'Top Geographic Drivers (Oct 2004 vs. Nov 2004)',
            rows: [
              { country: 'USA', previous: 198420.1, current: 394362.83, change: 195942.73, pct_change: 98.75 },
              { country: 'France', previous: 48210.5, current: 160628.9, change: 112418.4, pct_change: 233.18 },
              { country: 'Spain', previous: 62115.0, current: 138450.2, change: 76335.2, pct_change: 122.89 },
              { country: 'Australia', previous: 38900.0, current: 89120.4, change: 50220.4, pct_change: 129.1 },
              { country: 'Germany', previous: 42150.1, current: 13700.0, change: -28450.1, pct_change: -67.5 },
              { country: 'Sweden', previous: 32110.18, current: 0.0, change: -32110.18, pct_change: -100.0 },
              { country: 'Norway', previous: 41623.44, current: 0.0, change: -41623.44, pct_change: -100.0 },
            ],
          },
        ],
        business_report: {
          title: 'Executive Business Briefing: November 2004 Sales Performance & Root-Cause Audit',
          executive_summary: 'Independent ledger reconciliation confirms that November 2004 achieved the highest monthly gross merchandise revenue of the year at **$951,976.17** (**+72.07% MoM** over October 2004). Targeted pockets of contraction occurred in the **Ships** product line (**-$15,519.55**) and Nordic accounts (**-$73,733.62** across Norway and Sweden).',
          business_objective: 'Verify the premise of a November 2004 sales decline, decompose revenue variance across volume, AOV, product lines, and geographies, and test root-cause hypotheses.',
          kpi_overview: 'Nov 2004 Revenue: $951,976.17 | Oct 2004 Revenue: $553,257.50 | Net Variance: +$398,718.67 (+72.07%) | Orders: 33 vs. 13',
          sections: [
            { heading: 'Period-over-Period Reconciliation & Premise Verification', body: '- **Total Gross Revenue**: Increased from **$553,257.50** (Oct 2004) to **$951,976.17** (Nov 2004), a net gain of **+$398,718.67**.\n- **Accounting Integrity**: Sum of product line variances (**+$398,718.67**) exactly equals baseline net change (**$0.00** residual discrepancy).', evidence_ids: ['q_baseline_nov_oct', 'ev_py_rev_decomp'] },
            { heading: 'Dimensional Drivers & Localized Contractions', body: '- **Growth Engines**: **Classic Cars (+$156,743.11)** and **Vintage Cars (+$104,493.36)** contributed **65.52%** of the total monthly surge.\n- **Contraction Pockets**: **Ships (-$15,519.55 / -37.19%)** was the sole declining product line; **Norway (-$41,623.44)** and **Sweden (-$32,110.18)** had zero November orders after October bulk shipments.', evidence_ids: ['q_product_line_bridge', 'q_country_bridge'] },
            { heading: 'Statistical Price-Volume & Hypothesis Evaluation', body: '- **LMDI Decomposition**: Transaction volume expansion (**+20 orders / +153.85%**) was the dominant positive driver, while average order size normalized from **$42,558.27** to **$28,847.76**.\n- **Order Status Audit**: Hypothesis H3 (cancellation spike) was **rejected** with 0 cancelled orders in November 2004.', evidence_ids: ['ev_py_rev_decomp'] },
          ],
          recommendations: [
            'Investigate Ships product line inventory and distributor demand following its -$15,519.55 (-37.19%) counter-seasonal dip during the Q4 peak.',
            'Establish quarterly cohort tracking for Nordic B2B accounts (Norway, Sweden) whose bi-monthly ordering cadence creates artificial single-month drops.',
            'Protect Q4 fulfillment capacity for Classic Cars and Vintage Cars in USA and France, which accounted for 65.52% of November growth.',
          ],
          limitations: [
            'Product line and country bridges represent overlapping slices of the same $398,718.67 net variance and must not be summed together.',
            'Historical ClassicModels ledger covers 2003-01-06 through 2005-05-31; external marketing spend is not recorded in schema.',
          ],
        },
        recommendations: [
          'Investigate Ships product line inventory and distributor demand following its -$15,519.55 (-37.19%) counter-seasonal dip during the Q4 peak.',
          'Establish quarterly cohort tracking for Nordic B2B accounts (Norway, Sweden) whose bi-monthly ordering cadence creates artificial single-month drops.',
          'Protect Q4 fulfillment capacity for Classic Cars and Vintage Cars in USA and France, which accounted for 65.52% of November growth.',
        ],
        limitations: [
          'Product line and country bridges represent overlapping slices of the same $398,718.67 net variance and must not be summed together.',
          'Observational SQL & Python decomposition measures accounting contributions; causal attribution requires external campaign logs.',
        ],
        follow_up_questions: [
          'Compare 2004 vs 2003 revenue by product line',
          'Show the top 10 products by revenue in 2004',
          'Segment our B2B customers using RFM KMeans clustering and analyze Pareto revenue concentration',
        ],
        evidence_review: {
          round: 0,
          sufficient: true,
          decision: 'sufficient',
          notes: [
            '3-Layer Verification Passed: Python sandbox findings independently reproduced against immutable SHA-256 DatasetStore snapshots (0 mismatches, tolerance <= 1e-4).',
            'Product line bridge sum (+$398,718.67) reconciled exactly with baseline net variance (+$398,718.67).',
          ],
        },
        task_graph: [
          { id: 'baseline', title: 'Oct vs. Nov 2004 Baseline Revenue & Order Ledger', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 18 },
          { id: 'product_line_bridge', title: 'Product Line MoM Variance Bridge', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 22 },
          { id: 'country_bridge', title: 'Geographic Country MoM Variance Bridge', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 2, origin: 'planner', round: 0, duration_ms: 24 },
          { id: 'monthly_trend_2004', title: '2004 Monthly Revenue & Order Trajectory', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 2, origin: 'planner', round: 0, duration_ms: 19 },
          { id: 'rev_decomp_python', title: 'Autonomous Sandbox LMDI Volume vs. AOV & Concentration Decomposition', agent: 'python', status: 'COMPLETED', depends_on: ['baseline', 'product_line_bridge'], priority: 2, origin: 'planner', round: 0, duration_ms: 412 },
          { id: 'hypothesis_eval', title: 'Statistical Hypothesis Testing (Volume vs. Category vs. Status)', agent: 'hypothesis', status: 'COMPLETED', depends_on: ['baseline', 'product_line_bridge', 'country_bridge'], priority: 3, origin: 'planner', round: 0, duration_ms: 365 },
        ],
        python_analyses: [
          {
            task_id: 'rev_decomp_python',
            tool: 'autonomous_research',
            methodology: 'LMDI Logarithmic Price-Volume Decomposition & Bridge Reconciliation',
            verification_status: 'reproduced',
            repair_attempts: 0,
            duration_ms: 412,
            assumptions_checked: [
              'Verified zero missing or null revenue values across all 7 product lines',
              'Checked exact additivity: sum(product_line_deltas) == baseline_net_variance ($398,718.67)',
              'Applied logarithmic mean Divisia index (LMDI) to isolate pure order volume effect from AOV mix effect without residual interaction term',
            ],
            generated_code: `import math
import pandas as pd

base_df = load_dataset("baseline")
pl_df = load_dataset("product_line_bridge")

prev_rev, curr_rev = float(base_df["previous_revenue"].iloc[0]), float(base_df["current_revenue"].iloc[0])
prev_ord, curr_ord = float(base_df["previous_orders"].iloc[0]), float(base_df["current_orders"].iloc[0])
net_change = round(curr_rev - prev_rev, 2)

L_weight = (curr_rev - prev_rev) / (math.log(curr_rev) - math.log(prev_rev))
lmdi_order_volume_effect = round(L_weight * math.log(curr_ord / prev_ord), 2)
lmdi_aov_mix_effect = round(net_change - lmdi_order_volume_effect, 2)

pl_df["change"] = (pl_df["current"] - pl_df["previous"]).round(2)
bridge_sum = round(float(pl_df["change"].sum()), 2)
discrepancy = round(abs(net_change - bridge_sum), 4)

save_result({
    "previous_revenue_usd": round(prev_rev, 2),
    "current_revenue_usd": round(curr_rev, 2),
    "net_variance_usd": net_change,
    "pct_growth": round(((curr_rev - prev_rev) / prev_rev) * 100.0, 2),
    "lmdi_order_volume_effect_usd": lmdi_order_volume_effect,
    "lmdi_aov_effect_usd": lmdi_aov_mix_effect,
    "reconciliation_discrepancy_usd": discrepancy,
    "reconciled": bool(discrepancy <= 0.05),
}, records=pl_df.to_dict(orient="records"))`,
            execution_steps: [
              { step_index: 1, phase: 'analyze', action: 'run_python', thought: 'Decompose the Oct->Nov 2004 revenue change into exact LMDI order volume vs. AOV effects and verify that the 7 product-line deltas reconcile to the baseline net variance.', status: 'ok', duration_ms: 385, observation_summary: 'Sandbox COMPLETED: net_variance_usd=398718.67, lmdi_order_volume_effect_usd=679492.41, lmdi_aov_effect_usd=-280773.74, reconciliation_discrepancy_usd=0.0, reconciled=True' },
              { step_index: 2, phase: 'conclude', action: 'finish', thought: 'All accounting bridges reconcile with 0.00 discrepancy and LMDI decomposition cleanly separates order count growth from AOV normalization.', status: 'completed', duration_ms: 27, observation_summary: 'Concluded LMDI decomposition and verified independent SHA-256 dataset snapshot reproducibility.' },
            ],
            calculation: {
              tool: 'autonomous_research',
              methodology: 'LMDI Logarithmic Price-Volume Decomposition & Bridge Reconciliation',
              summary: { previous_revenue_usd: 553257.5, current_revenue_usd: 951976.17, net_variance_usd: 398718.67, pct_growth: 72.07, previous_aov_usd: 42558.27, current_aov_usd: 28847.76, lmdi_order_volume_effect_usd: 679492.41, lmdi_aov_effect_usd: -280773.74, reconciliation_discrepancy_usd: 0.0, reconciled: true },
            },
          },
        ],
        evidence: [
          {
            task_id: 'baseline',
            query_id: 'q_baseline_nov_oct',
            purpose: 'Compare total gross merchandise revenue and distinct order volume between Oct 2004 and Nov 2004',
            columns: ['previous_revenue', 'current_revenue', 'previous_orders', 'current_orders'],
            rows: [{ previous_revenue: 553257.5, current_revenue: 951976.17, previous_orders: 13, current_orders: 33 }],
            sql: `SELECT
  ROUND(SUM(CASE WHEN o.orderDate BETWEEN '2004-10-01' AND '2004-10-31' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue,
  ROUND(SUM(CASE WHEN o.orderDate BETWEEN '2004-11-01' AND '2004-11-30' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue,
  COUNT(DISTINCT CASE WHEN o.orderDate BETWEEN '2004-10-01' AND '2004-10-31' THEN o.orderNumber END) AS previous_orders,
  COUNT(DISTINCT CASE WHEN o.orderDate BETWEEN '2004-11-01' AND '2004-11-30' THEN o.orderNumber END) AS current_orders
FROM orders o
JOIN orderdetails od ON o.orderNumber = od.orderNumber
WHERE o.status NOT IN ('Cancelled') AND o.orderDate BETWEEN '2004-10-01' AND '2004-11-30'`,
            duration_ms: 18,
            dataset_id: 'ds_sha256_9a8b7c6d5e4f3a2b',
          },
          {
            task_id: 'product_line_bridge',
            query_id: 'q_product_line_bridge',
            purpose: 'Exhaustive product line variance breakdown between October 2004 and November 2004',
            columns: ['label', 'previous', 'current', 'change'],
            rows: [
              { label: 'Classic Cars', previous: 211450.25, current: 368193.36, change: 156743.11 },
              { label: 'Vintage Cars', previous: 118920.1, current: 223413.46, change: 104493.36 },
            ],
            sql: `SELECT p.productLine AS label,
  ROUND(SUM(CASE WHEN o.orderDate BETWEEN '2004-10-01' AND '2004-10-31' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous,
  ROUND(SUM(CASE WHEN o.orderDate BETWEEN '2004-11-01' AND '2004-11-30' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current
FROM orders o JOIN orderdetails od ON o.orderNumber = od.orderNumber JOIN products p ON od.productCode = p.productCode
WHERE o.status NOT IN ('Cancelled') AND o.orderDate BETWEEN '2004-10-01' AND '2004-11-30'
GROUP BY p.productLine`,
            duration_ms: 22,
            dataset_id: 'ds_sha256_4f3e2d1c0b9a8f7e',
          },
        ],
        models_used: [
          { agent_role: 'Jev Decision Router', task_type: 'decision_routing', model: 'typesafe/jev-1.14', reasoning_effort: 'decision_choice', duration_ms: 195, tokens: 412, cost_usd: 0.00021 },
          { agent_role: 'Lead Analyst / Planner', task_type: 'investigation_planning', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1140, tokens: 2840, cost_usd: 0.00568 },
          { agent_role: 'SQL Analyst (4 Parallel Queries)', task_type: 'sql_generation', model: 'gpt-6-luna', reasoning_effort: 'low', duration_ms: 620, tokens: 1920, cost_usd: 0.00115 },
          { agent_role: 'Python Research Agent', task_type: 'python_research', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1280, tokens: 3150, cost_usd: 0.0063 },
          { agent_role: 'Hypothesis Investigator', task_type: 'hypothesis_generation', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 980, tokens: 2210, cost_usd: 0.00442 },
          { agent_role: 'Evidence Reviewer (3-Layer Audit)', task_type: 'evidence_review', model: 'gpt-6.1-sol', reasoning_effort: 'low', duration_ms: 740, tokens: 2490, cost_usd: 0.00498 },
          { agent_role: 'Insight & Executive Report Writer', task_type: 'report_writing', model: 'gpt-6.1-terra', reasoning_effort: 'medium', duration_ms: 1320, tokens: 3680, cost_usd: 0.00552 },
        ],
        trace: [
          { stage: 'prepare', message: "Resolved metric: 'gross_revenue', period: '2004-11 vs 2004-10', tables: 8 authorized", status: 'ok' },
          { stage: 'jev_route', message: 'Jev route: ROOT_CAUSE_INVESTIGATION (confidence: 0.97)', status: 'ok' },
          { stage: 'plan', message: 'Planning Agent drafted 6 tasks (complexity=investigation, hypotheses=True)', status: 'ok' },
          { stage: 'supervisor', message: 'Dispatching Wave 1 ready SQL tasks in parallel: baseline, product_line_bridge, country_bridge, monthly_trend_2004', status: 'ok' },
          { stage: 'python_research_agent_rev_decomp_python', message: 'Step 1 (rev_decomp_python): executed LMDI decomposition & bridge verification in isolated sandbox (385 ms)', status: 'ok' },
          { stage: 'review_evidence', message: '3-Layer Evidence Reviewer reproduced sandbox Python code against SHA-256 dataset snapshots (decision=sufficient)', status: 'ok' },
          { stage: 'quality_gate', message: 'Final Quality Gate verified all 18 currency/metric figures in narrative against source evidence', status: 'ok' },
        ],
      },
    },
  },
  {
    id: 'demo-rfm-kmeans-sandbox',
    badge: 'Autonomous Python Sandbox (scikit-learn)',
    title: 'Customer RFM KMeans Clustering & Pareto Revenue Concentration',
    question: 'Segment our B2B customers using RFM KMeans clustering and analyze Pareto revenue concentration',
    description: 'Showcases open-ended Python research: dynamic scikit-learn KMeans + silhouette scoring, Gini/Pareto concentration, and SHA-256 reproduction.',
    response: {
      run_id: 'demo-rfm-kmeans-2005',
      session_id: 'demo-session-showcase',
      route: 'COMPLEX_ANALYSIS',
      route_confidence: 0.95,
      route_probabilities: { COMPLEX_ANALYSIS: 0.95, ROOT_CAUSE_INVESTIGATION: 0.04, SIMPLE_SQL: 0.01 },
      report: {
        title: 'ClassicModels B2B Customer RFM KMeans Segmentation & Pareto Concentration',
        summary: 'Autonomous Python sandbox execution segmented all **98 active ClassicModels B2B customers** across **$9,604,190.61** in cumulative gross revenue using Z-score standardized **RFM KMeans clustering ($k=3$, silhouette score = 0.5842)**. A **VIP Anchor Tier** of just **2 customers** (**Euro+ Shopping Channel** at **$820,689.54** and **Mini Gifts Distributors Ltd.** at **$591,827.34**) generates **$1,412,516.88 (14.71% of total company revenue)** with an average of **21.5 orders** each. Combined with the **Core Repeat Tier (44 customers, $4,918,345.19 / 51.21%)**, the top 20% of customers account for **46.84% of revenue** (Gini coefficient = **0.3918**).',
        status: 'complete',
        output_mode: 'answer_with_visuals',
        period: 'Full Historical Ledger (2003-01 to 2005-05)',
        comparison: 'Cross-Cluster Cohort Benchmark',
        kpis: [
          { label: 'Active B2B Customers', value: 98, unit: 'customers' },
          { label: 'Total Ledger Revenue', value: 9604190.61, unit: 'USD' },
          { label: 'Top 20% Pareto Share', value: 46.84, unit: '%' },
          { label: 'RFM Silhouette Score', value: 0.5842, unit: 'score' },
          { label: 'VIP Anchor Tier Revenue', value: 1412516.88, unit: 'USD' },
          { label: 'Revenue Gini Coefficient', value: 0.3918, unit: 'index' },
        ],
        findings: [
          { title: '3-Cluster RFM Separation Verified via Silhouette & Kruskal-Wallis Tests', detail: 'Standardized Recency (days since last order relative to `2005-05-31`), Frequency (distinct order count), and Monetary value separated cleanly into 3 clusters ($k=3$, silhouette = **0.5842**, Kruskal-Wallis $p < 0.0001$ across all 3 features).', status: 'verified', evidence_ids: ['q_customer_rfm_ledger', 'ev_py_rfm_kmeans'] },
          { title: 'Extreme Top-2 Account Concentration Risk (14.71% of Total Revenue)', detail: '**Euro+ Shopping Channel** (**$820,689.54**, 26 orders, 0 days recency) and **Mini Gifts Distributors Ltd.** (**$591,827.34**, 17 orders, 2 days recency) form a standalone **VIP Anchor Tier** generating **$1,412,516.88**—nearly 7x the average customer lifetime value (**$98,001.94**).', status: 'verified', evidence_ids: ['q_customer_rfm_ledger', 'ev_py_rfm_kmeans'] },
          { title: 'At-Risk / Lapsed B2B Tier (52 Customers, 264.8 Avg Recency Days)', detail: '**52 customers (53.06% of customer base)** fall into the **Low-Frequency / Lapsed Tier**, averaging **2.15 orders**, **$62,948.63** cumulative spend, and **264.8 days** since their last order—representing **$3,273,328.54 (34.08%)** in at-risk repeat revenue.', status: 'verified', evidence_ids: ['ev_py_rfm_kmeans'] },
        ],
        hypotheses: [],
        charts: [
          { title: 'Total Revenue by RFM Customer Cluster (USD)', kind: 'bar', labels: ['Core Repeat B2B (44 Customers)', 'At-Risk / Low-Freq (52 Customers)', 'VIP Anchor Accounts (2 Customers)'], values: [4918345.19, 3273328.54, 1412516.88], unit: 'USD', x_label: 'Cluster Revenue (USD)', y_label: 'RFM Segment', source_task_id: 'rfm_kmeans_python' },
          { title: 'Top 8 B2B Customers by Cumulative Gross Revenue', kind: 'bar', labels: ['Euro+ Shopping Channel', 'Mini Gifts Distributors Ltd.', 'Australian Collectors, Co.', 'Muscle Machine Inc', 'La Rochelle Gifts', 'Dragon Souveniers, Ltd.', 'Down Under Souveniers, Inc', 'Land of Toys Inc.'], values: [820689.54, 591827.34, 180585.07, 177913.95, 158573.12, 156251.03, 154622.08, 149085.15], unit: 'USD', x_label: 'Cumulative Gross Revenue (USD)', y_label: 'Customer', source_task_id: 'customer_rfm_sql' },
        ],
        tables: [
          {
            title: 'RFM KMeans Cluster Profiles (StandardScaler + KMeans k=3)',
            rows: [
              { cluster_segment: 'VIP Anchor Accounts', customers: 2, avg_recency_days: 1.0, avg_frequency_orders: 21.5, avg_monetary_usd: 706258.44, total_revenue_usd: 1412516.88, revenue_share_pct: 14.71 },
              { cluster_segment: 'Core Repeat B2B', customers: 44, avg_recency_days: 92.4, avg_frequency_orders: 3.84, avg_monetary_usd: 111780.57, total_revenue_usd: 4918345.19, revenue_share_pct: 51.21 },
              { cluster_segment: 'At-Risk / Low-Frequency', customers: 52, avg_recency_days: 264.8, avg_frequency_orders: 2.15, avg_monetary_usd: 62948.63, total_revenue_usd: 3273328.54, revenue_share_pct: 34.08 },
            ],
          },
        ],
        recommendations: [
          'Assign dedicated executive account coverage and multi-year contract protections to Euro+ Shopping Channel and Mini Gifts Distributors Ltd. (14.71% of company revenue).',
          'Launch an automated 180-day reactivation playbook targeting the 52 Lapsed/Low-Frequency accounts ($3.27M historical spend) before recency exceeds 12 months.',
          'Upsell high-margin Classic Cars and Vintage Cars bundles to the 44 Core Repeat B2B accounts to lift their order frequency from 3.84 toward 6+ orders.',
        ],
        limitations: [
          'Recency is anchored to the maximum ClassicModels ledger date (2005-05-31).',
          '24 catalog customers have 0 historical orders and are excluded from active RFM clustering (98 active out of 122 total customers).',
        ],
        follow_up_questions: [
          'Why did sales decline in November 2004 compared with October 2004?',
          'Show the top 10 products by revenue in 2004',
          'Compare 2004 vs 2003 revenue by product line',
        ],
        evidence_review: {
          round: 0,
          sufficient: true,
          decision: 'sufficient',
          notes: ['Independent Sandbox Reproduction Verified: Re-running generated scikit-learn KMeans code (random_state=42) against SHA-256 snapshot ds_sha256_11c2d3e4f5a6b7c8 reproduced silhouette_score=0.5842 and cluster revenue totals to 100% precision.'],
        },
        task_graph: [
          { id: 'customer_rfm_sql', title: 'Extract Customer-Level Recency, Frequency & Monetary Ledger', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 29 },
          { id: 'rfm_kmeans_python', title: 'Autonomous scikit-learn RFM KMeans Clustering & Lorenz Gini Analysis', agent: 'python', status: 'COMPLETED', depends_on: ['customer_rfm_sql'], priority: 2, origin: 'planner', round: 0, duration_ms: 615 },
        ],
        python_analyses: [
          {
            task_id: 'rfm_kmeans_python',
            tool: 'autonomous_research',
            methodology: 'Z-Score Standardized RFM KMeans (k=3) + Silhouette & Gini Concentration',
            verification_status: 'reproduced',
            repair_attempts: 0,
            duration_ms: 615,
            assumptions_checked: [
              'Applied StandardScaler Z-score normalization to prevent high-magnitude Monetary USD values from dominating Euclidean KMeans distances',
              'Fixed random_state=42 and n_init=10 for deterministic reproduction in EvidenceReviewerAgent',
              'Verified Kruskal-Wallis non-parametric separation across clusters (p < 0.0001)',
            ],
            generated_code: `import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = load_dataset("customer_rfm_sql")
scaled = StandardScaler().fit_transform(df[["recency_days", "frequency", "monetary"]].astype(float).values)
df["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)
sil = float(silhouette_score(scaled, df["cluster"]))
_, kw_p = stats.kruskal(*[g["monetary"].values for _, g in df.groupby("cluster")])

df_sorted = df.sort_values("monetary", ascending=False).reset_index(drop=True)
total_rev = float(df_sorted["monetary"].sum())
top_20_share = (float(df_sorted.iloc[:max(1, int(round(len(df_sorted) * 0.20)))]["monetary"].sum()) / total_rev) * 100.0
asc = np.sort(df["monetary"].astype(float).values)
n = len(asc)
gini = float((2.0 * np.sum(np.arange(1, n + 1) * asc)) / (n * np.sum(asc)) - (n + 1.0) / n)

save_result({
    "active_customers": int(len(df)), "total_revenue_usd": round(total_rev, 2),
    "n_clusters": 3, "silhouette_score": round(sil, 4),
    "kruskal_p_value": round(float(kw_p), 6),
    "top_20_pct_revenue_share": round(top_20_share, 2), "gini_coefficient": round(gini, 4),
})`,
            execution_steps: [
              { step_index: 1, phase: 'analyze', action: 'run_python', thought: 'Standardize recency_days, frequency, and monetary columns via StandardScaler, fit KMeans(k=3, random_state=42), evaluate silhouette score, and compute Pareto top-20% concentration and Gini index.', status: 'ok', duration_ms: 584, observation_summary: 'Sandbox COMPLETED: active_customers=98, silhouette_score=0.5842, top_20_pct_revenue_share=46.84, gini_coefficient=0.3918' },
              { step_index: 2, phase: 'conclude', action: 'finish', thought: 'Clustering separation and Pareto metrics are verified and reproducible.', status: 'completed', duration_ms: 31, observation_summary: 'Concluded RFM KMeans and Pareto analysis.' },
            ],
            calculation: {
              tool: 'autonomous_research',
              methodology: 'Z-Score Standardized RFM KMeans (k=3) + Silhouette & Gini Concentration',
              summary: { active_customers: 98, total_revenue_usd: 9604190.61, n_clusters: 3, silhouette_score: 0.5842, kruskal_p_value: 0.000012, top_20_pct_revenue_share: 46.84, gini_coefficient: 0.3918 },
            },
          },
        ],
        evidence: [
          {
            task_id: 'customer_rfm_sql',
            query_id: 'q_customer_rfm_ledger',
            purpose: 'Compute customer-level Recency (days since last order to 2005-05-31), Frequency (distinct orders), and Monetary gross revenue',
            columns: ['customerNumber', 'customerName', 'country', 'recency_days', 'frequency', 'monetary'],
            rows: [
              { customerNumber: 141, customerName: 'Euro+ Shopping Channel', country: 'Spain', recency_days: 0, frequency: 26, monetary: 820689.54 },
              { customerNumber: 124, customerName: 'Mini Gifts Distributors Ltd.', country: 'USA', recency_days: 2, frequency: 17, monetary: 591827.34 },
            ],
            sql: `SELECT c.customerNumber, c.customerName, c.country,
  DATEDIFF('2005-05-31', MAX(o.orderDate)) AS recency_days,
  COUNT(DISTINCT o.orderNumber) AS frequency,
  ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS monetary
FROM customers c JOIN orders o ON c.customerNumber = o.customerNumber JOIN orderdetails od ON o.orderNumber = od.orderNumber
WHERE o.status NOT IN ('Cancelled')
GROUP BY c.customerNumber, c.customerName, c.country ORDER BY monetary DESC`,
            duration_ms: 29,
            dataset_id: 'ds_sha256_11c2d3e4f5a6b7c8',
          },
        ],
        models_used: [
          { agent_role: 'Jev Decision Router', task_type: 'decision_routing', model: 'typesafe/jev-1.14', reasoning_effort: 'decision_choice', duration_ms: 180, tokens: 390, cost_usd: 0.00019 },
          { agent_role: 'Lead Analyst / Planner', task_type: 'investigation_planning', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 940, tokens: 2150, cost_usd: 0.0043 },
          { agent_role: 'Autonomous SQL Research Agent', task_type: 'sql_research', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 680, tokens: 1640, cost_usd: 0.00328 },
          { agent_role: 'Python Research Agent (Sandbox)', task_type: 'python_research', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1490, tokens: 3420, cost_usd: 0.00684 },
          { agent_role: 'Evidence Reviewer (SHA-256 Repro)', task_type: 'evidence_review', model: 'gpt-6.1-sol', reasoning_effort: 'low', duration_ms: 620, tokens: 1890, cost_usd: 0.00378 },
          { agent_role: 'Insight Agent', task_type: 'insight_generation', model: 'gpt-6.1-terra', reasoning_effort: 'medium', duration_ms: 910, tokens: 2410, cost_usd: 0.00361 },
        ],
        trace: [
          { stage: 'prepare', message: "Resolved metric: 'gross_revenue', dynamic schema catalog loaded (8 tables)", status: 'ok' },
          { stage: 'jev_route', message: 'Jev route: COMPLEX_ANALYSIS (confidence: 0.95)', status: 'ok' },
          { stage: 'plan', message: 'Planning Agent drafted 2-node autonomous SQL + Python KMeans research DAG', status: 'ok' },
          { stage: 'exec_customer_rfm_sql', message: 'Fetched 98 active B2B customer RFM records in 29 ms (SHA-256 snapshot stored)', status: 'ok' },
          { stage: 'python_research_agent_rfm_kmeans_python', message: 'Step 1: executed StandardScaler + KMeans(k=3) + Kruskal-Wallis + Gini in isolated process sandbox (584 ms)', status: 'ok' },
          { stage: 'review_evidence', message: 'Evidence Reviewer independently reproduced KMeans silhouette=0.5842 in fresh sandbox workspace', status: 'ok' },
        ],
      },
    },
  },
];

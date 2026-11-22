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
    description: 'Demonstrates Jev routing, parallel SQL + Python + Hypothesis DAG, accounting bridge reconciliation, and SHA-256 sandbox reproduction.',
    response: {
      run_id: 'demo-rca-2004-11',
      session_id: 'demo-session-showcase',
      route: 'ROOT_CAUSE_INVESTIGATION',
      route_confidence: 0.97,
      route_probabilities: { ROOT_CAUSE_INVESTIGATION: 0.97, PERIOD_COMPARISON: 0.02, SIMPLE_SQL: 0.01 },
      report: {
        title: 'ClassicModels Gross Merchandise Revenue Root-Cause Analysis (Nov 2004 vs. Oct 2004)',
        summary: 'ClassicModels gross merchandise revenue **increased by +$398,718.67 (+72.07%)** from **$553,257.50** in October 2004 (13 orders) to **$951,976.17** in November 2004 (33 orders). Isolated declines occurred in **Ships (-$15,519.55)** and **Norway (-$41,623.44)**, while overall growth was driven by **+153.85% order volume expansion** (+20 orders), led by **Classic Cars (+$156,743.11)**, **Vintage Cars (+$104,493.36)**, and **USA (+$195,942.73)**.',
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
          { title: 'Baseline Accounting Reconciliation', detail: 'Gross merchandise revenue expanded from **$553,257.50** in October 2004 to **$951,976.17** in November 2004 (**+$398,718.67 / +72.07%**). Dimensional bridges reconciled to **$0.00 discrepancy**.', status: 'verified', evidence_ids: ['q_baseline_nov_oct'] },
          { title: 'Order Volume Expansion Outweighed AOV Compression', detail: 'Order count surged from **13 to 33 orders (+153.85%)**, contributing **+$679,492.41** in LMDI volume effect, partially offset by **-$280,773.74** AOV mix normalization.', status: 'verified', evidence_ids: ['ev_py_rev_decomp'] },
          { title: 'Category Pocket of Decline: Ships (-$15,519.55)', detail: '6 of 7 product lines grew—led by **Classic Cars (+$156,743.11)** and **Vintage Cars (+$104,493.36)**—while **Ships** contracted by **-$15,519.55 (-37.19%)**.', status: 'verified', evidence_ids: ['q_product_line_bridge'] },
        ],
        hypotheses: [
          { id: 'hyp_order_volume', description: 'H1: Order transaction volume change (13 vs. 33 orders) is the primary statistical driver of the monthly revenue shift.', dimension: 'order_volume', status: 'supported', measured_effect: 398718.67, share_of_change: 100.0, confidence: 0.96, evidence_query_ids: ['q_baseline_nov_oct'], limitations: 'Observational accounting decomposition.' },
          { id: 'hyp_ships_contraction', description: 'H2: Isolated contraction in Ships (-$15,519.55) and Nordic accounts (-$73,733.62) created localized sales drops.', dimension: 'product_line', status: 'supported', measured_effect: -15519.55, share_of_change: -3.89, confidence: 0.94, evidence_query_ids: ['q_product_line_bridge'], limitations: 'Product line and country dimensions overlap.' },
          { id: 'hyp_cancellation_spike', description: 'H3: Order cancellations in November 2004 caused an overall revenue decline.', dimension: 'order_status', status: 'rejected', measured_effect: 0.0, share_of_change: 0.0, confidence: 0.99, evidence_query_ids: ['q_baseline_nov_oct'], limitations: 'Rejected: 0 cancelled orders in November 2004.' },
        ],
        charts: [
          { title: 'Product Line Revenue Variance (Nov 2004 vs. Oct 2004)', kind: 'bar', labels: ['Classic Cars', 'Vintage Cars', 'Motorcycles', 'Planes', 'Trucks and Buses', 'Trains', 'Ships'], values: [156743.11, 104493.36, 68914.22, 46812.4, 31458.95, 5816.18, -15519.55], unit: 'USD' },
          { title: '2004 Monthly Gross Merchandise Revenue Trajectory', kind: 'line', labels: ['2004-01', '2004-02', '2004-03', '2004-04', '2004-05', '2004-06', '2004-07', '2004-08', '2004-09', '2004-10', '2004-11', '2004-12'], values: [292385.21, 289502.84, 217691.26, 187575.77, 248325.3, 343371.0, 325563.49, 419327.09, 283799.8, 553257.5, 951976.17, 428582.46], unit: 'USD' },
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
        ],
        recommendations: [
          'Investigate Ships product line inventory and distributor demand following its -$15,519.55 (-37.19%) dip during the Q4 peak.',
          'Establish quarterly cohort tracking for Nordic B2B accounts (Norway, Sweden) whose bi-monthly ordering cadence creates single-month drops.',
          'Protect Q4 fulfillment capacity for Classic Cars and Vintage Cars in USA and France (65.52% of November growth).',
        ],
        limitations: [
          'Product line and country bridges represent overlapping slices of the same $398,718.67 net variance.',
          'Observational decomposition measures accounting contributions rather than external causal triggers.',
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
          notes: ['3-Layer Verification Passed: Python sandbox findings independently reproduced against SHA-256 DatasetStore snapshots.'],
        },
        task_graph: [
          { id: 'baseline', title: 'Oct vs. Nov 2004 Baseline Revenue & Order Ledger', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 18 },
          { id: 'product_line_bridge', title: 'Product Line MoM Variance Bridge', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 22 },
          { id: 'rev_decomp_python', title: 'Sandbox LMDI Volume vs. AOV Decomposition', agent: 'python', status: 'COMPLETED', depends_on: ['baseline', 'product_line_bridge'], priority: 2, origin: 'planner', round: 0, duration_ms: 412 },
          { id: 'hypothesis_eval', title: 'Statistical Hypothesis Testing', agent: 'hypothesis', status: 'COMPLETED', depends_on: ['baseline', 'product_line_bridge'], priority: 3, origin: 'planner', round: 0, duration_ms: 365 },
        ],
        python_analyses: [
          {
            task_id: 'rev_decomp_python',
            tool: 'autonomous_research',
            methodology: 'LMDI Logarithmic Price-Volume Decomposition & Bridge Reconciliation',
            verification_status: 'reproduced',
            repair_attempts: 0,
            duration_ms: 412,
            assumptions_checked: ['Verified zero missing values across all 7 product lines', 'Checked exact additivity against baseline net variance ($398,718.67)'],
            generated_code: `import math\nbase_df = load_dataset("baseline")\npl_df = load_dataset("product_line_bridge")\nprev_rev, curr_rev = float(base_df["previous_revenue"].iloc[0]), float(base_df["current_revenue"].iloc[0])\nprev_ord, curr_ord = float(base_df["previous_orders"].iloc[0]), float(base_df["current_orders"].iloc[0])\nnet_change = round(curr_rev - prev_rev, 2)\nL = (curr_rev - prev_rev) / (math.log(curr_rev) - math.log(prev_rev))\nvol_eff = round(L * math.log(curr_ord / prev_ord), 2)\nsave_result({"net_variance_usd": net_change, "lmdi_order_volume_effect_usd": vol_eff, "lmdi_aov_effect_usd": round(net_change - vol_eff, 2), "reconciled": True})`,
            execution_steps: [
              { step_index: 1, phase: 'analyze', action: 'run_python', thought: 'Decompose Oct->Nov 2004 revenue change into LMDI volume vs. AOV effects.', status: 'ok', duration_ms: 385, observation_summary: 'net_variance_usd=398718.67, lmdi_order_volume_effect_usd=679492.41, reconciled=True' },
            ],
            calculation: {
              tool: 'autonomous_research',
              methodology: 'LMDI Logarithmic Price-Volume Decomposition & Bridge Reconciliation',
              summary: { previous_revenue_usd: 553257.5, current_revenue_usd: 951976.17, net_variance_usd: 398718.67, lmdi_order_volume_effect_usd: 679492.41, lmdi_aov_effect_usd: -280773.74, reconciled: true },
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
            sql: "SELECT ROUND(SUM(CASE WHEN o.orderDate < '2004-11-01' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS previous_revenue, ROUND(SUM(CASE WHEN o.orderDate >= '2004-11-01' THEN od.quantityOrdered * od.priceEach ELSE 0 END), 2) AS current_revenue FROM orders o JOIN orderdetails od ON o.orderNumber = od.orderNumber WHERE o.status != 'Cancelled' AND o.orderDate >= '2004-10-01' AND o.orderDate < '2004-12-01'",
            duration_ms: 18,
            dataset_id: 'ds_sha256_9a8b7c6d5e4f3a2b',
          },
        ],
        models_used: [
          { agent_role: 'Jev Decision Router', task_type: 'decision_routing', model: 'typesafe/jev-1.14', reasoning_effort: 'decision_choice', duration_ms: 195, tokens: 412, cost_usd: 0.00021 },
          { agent_role: 'Planning Agent', task_type: 'investigation_planning', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1140, tokens: 2840, cost_usd: 0.00568 },
          { agent_role: 'Python Research Agent', task_type: 'python_research', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1280, tokens: 3150, cost_usd: 0.0063 },
          { agent_role: 'Report Generation Agent', task_type: 'report_writing', model: 'gpt-6.1-terra', reasoning_effort: 'low', duration_ms: 1320, tokens: 3680, cost_usd: 0.00552 },
        ],
        trace: [
          { stage: 'prepare', message: "Resolved metric: 'gross_revenue', period: '2004-11'", status: 'ok' },
          { stage: 'jev_route', message: 'Jev route: ROOT_CAUSE_INVESTIGATION (confidence: 0.97)', status: 'ok' },
          { stage: 'supervisor', message: 'Dispatching ready tasks in parallel: baseline, product_line_bridge', status: 'ok' },
          { stage: 'python_rev_decomp_python', message: 'Completed autonomous Python LMDI decomposition (385 ms)', status: 'ok' },
          { stage: 'quality_gate', message: 'Final Quality Gate passed (status=complete)', status: 'ok' },
        ],
      },
    },
  },
  {
    id: 'demo-rfm-kmeans-sandbox',
    badge: 'Autonomous Python Sandbox (scikit-learn)',
    title: 'Customer RFM KMeans Clustering & Pareto Revenue Concentration',
    question: 'Segment our B2B customers using RFM KMeans clustering and analyze Pareto revenue concentration',
    description: 'Showcases open-ended Python research: scikit-learn KMeans + silhouette scoring, Gini/Pareto concentration, and SHA-256 reproduction.',
    response: {
      run_id: 'demo-rfm-kmeans-2005',
      session_id: 'demo-session-showcase',
      route: 'COMPLEX_ANALYSIS',
      route_confidence: 0.95,
      route_probabilities: { COMPLEX_ANALYSIS: 0.95, ROOT_CAUSE_INVESTIGATION: 0.04, SIMPLE_SQL: 0.01 },
      report: {
        title: 'ClassicModels B2B Customer RFM KMeans Segmentation & Pareto Concentration',
        summary: 'Autonomous Python sandbox execution segmented all **98 active ClassicModels B2B customers** across **$9,604,190.61** in cumulative gross revenue using Z-score standardized **RFM KMeans clustering ($k=3$, silhouette score = 0.5842)**. A **VIP Anchor Tier** of **2 customers** (**Euro+ Shopping Channel** and **Mini Gifts Distributors Ltd.**) generates **$1,412,516.88 (14.71% of total revenue)**, while the top 20% of customers account for **46.84% of revenue** (Gini = **0.3918**).',
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
          { title: '3-Cluster RFM Separation Verified via Silhouette & Kruskal-Wallis Tests', detail: 'Standardized Recency, Frequency, and Monetary value separated cleanly into 3 clusters ($k=3$, silhouette = **0.5842**, Kruskal-Wallis $p < 0.0001$).', status: 'verified', evidence_ids: ['q_customer_rfm_ledger'] },
          { title: 'Top-2 Account Concentration (14.71% of Total Revenue)', detail: '**Euro+ Shopping Channel** (**$820,689.54**, 26 orders) and **Mini Gifts Distributors Ltd.** (**$591,827.34**, 17 orders) form a standalone **VIP Anchor Tier** generating **$1,412,516.88**.', status: 'verified', evidence_ids: ['q_customer_rfm_ledger'] },
          { title: 'At-Risk / Lapsed B2B Tier (52 Customers)', detail: '**52 customers (53.06%)** fall into the **Low-Frequency / Lapsed Tier**, averaging **2.15 orders** and **264.8 days** since their last order (**$3,273,328.54** cumulative spend).', status: 'verified', evidence_ids: ['ev_py_rfm_kmeans'] },
        ],
        hypotheses: [],
        charts: [
          { title: 'Total Revenue by RFM Customer Cluster (USD)', kind: 'bar', labels: ['Core Repeat B2B (44 Customers)', 'At-Risk / Low-Freq (52 Customers)', 'VIP Anchor Accounts (2 Customers)'], values: [4918345.19, 3273328.54, 1412516.88], unit: 'USD' },
          { title: 'Top 8 B2B Customers by Cumulative Gross Revenue', kind: 'bar', labels: ['Euro+ Shopping Channel', 'Mini Gifts Distributors Ltd.', 'Australian Collectors, Co.', 'Muscle Machine Inc', 'La Rochelle Gifts', 'Dragon Souveniers, Ltd.', 'Down Under Souveniers, Inc', 'Land of Toys Inc.'], values: [820689.54, 591827.34, 180585.07, 177913.95, 158573.12, 156251.03, 154622.08, 149085.15], unit: 'USD' },
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
          'Assign dedicated executive account coverage to Euro+ Shopping Channel and Mini Gifts Distributors Ltd. (14.71% of revenue).',
          'Launch a 180-day reactivation playbook targeting the 52 Lapsed/Low-Frequency accounts ($3.27M historical spend).',
          'Upsell Classic Cars and Vintage Cars bundles to the 44 Core Repeat B2B accounts.',
        ],
        limitations: [
          'Recency is anchored to the maximum ClassicModels ledger date (2005-05-31).',
          '24 catalog customers with 0 historical orders are excluded from active RFM clustering.',
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
          notes: ['Independent Sandbox Reproduction Verified: Re-running generated scikit-learn KMeans code reproduced silhouette_score=0.5842 and cluster totals.'],
        },
        task_graph: [
          { id: 'customer_rfm_sql', title: 'Extract Customer-Level Recency, Frequency & Monetary Ledger', agent: 'sql', status: 'COMPLETED', depends_on: [], priority: 1, origin: 'planner', round: 0, duration_ms: 29 },
          { id: 'rfm_kmeans_python', title: 'Autonomous scikit-learn RFM KMeans Clustering & Gini Analysis', agent: 'python', status: 'COMPLETED', depends_on: ['customer_rfm_sql'], priority: 2, origin: 'planner', round: 0, duration_ms: 615 },
        ],
        python_analyses: [
          {
            task_id: 'rfm_kmeans_python',
            tool: 'autonomous_research',
            methodology: 'Z-Score Standardized RFM KMeans (k=3) + Silhouette & Gini Concentration',
            verification_status: 'reproduced',
            repair_attempts: 0,
            duration_ms: 615,
            assumptions_checked: ['Applied StandardScaler Z-score normalization', 'Fixed random_state=42 for deterministic reproduction'],
            generated_code: `from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nfrom sklearn.preprocessing import StandardScaler\ndf = load_dataset("customer_rfm_sql")\nscaled = StandardScaler().fit_transform(df[["recency_days", "frequency", "monetary"]].astype(float).values)\ndf["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)\nsave_result({"active_customers": len(df), "silhouette_score": round(float(silhouette_score(scaled, df["cluster"])), 4)})`,
            execution_steps: [
              { step_index: 1, phase: 'analyze', action: 'run_python', thought: 'Standardize RFM columns, fit KMeans(k=3), and compute silhouette score and Gini index.', status: 'ok', duration_ms: 584, observation_summary: 'active_customers=98, silhouette_score=0.5842, top_20_pct_revenue_share=46.84' },
            ],
            calculation: {
              tool: 'autonomous_research',
              methodology: 'Z-Score Standardized RFM KMeans (k=3) + Silhouette & Gini Concentration',
              summary: { active_customers: 98, total_revenue_usd: 9604190.61, n_clusters: 3, silhouette_score: 0.5842, top_20_pct_revenue_share: 46.84, gini_coefficient: 0.3918 },
            },
          },
        ],
        evidence: [
          {
            task_id: 'customer_rfm_sql',
            query_id: 'q_customer_rfm_ledger',
            purpose: 'Compute customer-level Recency, Frequency, and Monetary gross revenue',
            columns: ['customerNumber', 'customerName', 'country', 'recency_days', 'frequency', 'monetary'],
            rows: [
              { customerNumber: 141, customerName: 'Euro+ Shopping Channel', country: 'Spain', recency_days: 0, frequency: 26, monetary: 820689.54 },
              { customerNumber: 124, customerName: 'Mini Gifts Distributors Ltd.', country: 'USA', recency_days: 2, frequency: 17, monetary: 591827.34 },
            ],
            sql: "SELECT c.customerNumber, c.customerName, c.country, DATEDIFF('2005-05-31', MAX(o.orderDate)) AS recency_days, COUNT(DISTINCT o.orderNumber) AS frequency, ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS monetary FROM customers c JOIN orders o ON c.customerNumber = o.customerNumber JOIN orderdetails od ON o.orderNumber = od.orderNumber WHERE o.status != 'Cancelled' GROUP BY c.customerNumber, c.customerName, c.country ORDER BY monetary DESC",
            duration_ms: 29,
            dataset_id: 'ds_sha256_11c2d3e4f5a6b7c8',
          },
        ],
        models_used: [
          { agent_role: 'Jev Decision Router', task_type: 'decision_routing', model: 'typesafe/jev-1.14', reasoning_effort: 'decision_choice', duration_ms: 180, tokens: 390, cost_usd: 0.00019 },
          { agent_role: 'Planning Agent', task_type: 'investigation_planning', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 940, tokens: 2150, cost_usd: 0.0043 },
          { agent_role: 'Python Research Agent', task_type: 'python_research', model: 'gpt-6.1-sol', reasoning_effort: 'medium', duration_ms: 1490, tokens: 3420, cost_usd: 0.00684 },
        ],
        trace: [
          { stage: 'prepare', message: "Resolved metric: 'gross_revenue', loaded 8 tables", status: 'ok' },
          { stage: 'jev_route', message: 'Jev route: COMPLEX_ANALYSIS (confidence: 0.95)', status: 'ok' },
          { stage: 'exec_customer_rfm_sql', message: 'Fetched 98 active B2B customer RFM records in 29 ms', status: 'ok' },
          { stage: 'python_rfm_kmeans_python', message: 'Executed StandardScaler + KMeans(k=3) in sandbox (584 ms)', status: 'ok' },
        ],
      },
    },
  },
];

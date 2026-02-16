# Agentic AI — Demand Blocking & Smart Rebooking

> **GCC FUSIONX — Problem Statement 4**
> An autonomous, production-ready Agentic AI system that predicts demand spikes, proactively blocks inventory, and recovers margin through smart rebooking.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Database Configuration](#database-configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Agent Descriptions](#agent-descriptions)
- [ML Models](#ml-models)
- [Business Logic](#business-logic)
- [Success Metrics](#success-metrics)

---

## Overview

This system implements a **7-agent autonomous pipeline** that:

1. **Predicts the future** — demand spikes, sell-outs, price changes using XGBoost/LightGBM models trained on real database data
2. **Takes proactive action** — blocks inventory before competition using multi-criteria scoring
3. **Optimizes continuously** — recovers margin via risk-aware smart rebooking
4. **Reports transparently** — generates weekly executive reports with KPIs and AI-driven recommendations

### Key Principles
- **NO demo/synthetic data** — all predictions from ML models trained on real MS SQL data
- **NO hardcoded values** — all thresholds and parameters are data-driven
- **Fully explainable** — every decision includes clear reasoning
- **Production-ready** — proper error handling, logging, database transactions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENTIC AI SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   SENSE      │  │   PREDICT    │  │   DECIDE     │      │
│  │  Agent       │→ │   Agent      │→ │   Agent      │      │
│  │              │  │              │  │              │      │
│  │ • Data       │  │ • Demand     │  │ • Blocking   │      │
│  │   Ingestion  │  │   Spike      │  │   Strategy   │      │
│  │ • Feature    │  │ • Sellout    │  │ • Rebooking  │      │
│  │   Engineer   │  │   Risk       │  │   Logic      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                  ↓              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  RESERVE     │  │   MONITOR    │  │  OPTIMIZE    │      │
│  │  Agent       │  │   Agent      │  │  Agent       │      │
│  │              │  │              │  │              │      │
│  │ • Execute    │  │ • Track      │  │ • Rebook     │      │
│  │   Blocks     │  │   Bookings   │  │   Execute    │      │
│  │ • Confirm    │  │ • Price      │  │ • Margin     │      │
│  │   Actions    │  │   Watch      │  │   Recovery   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                              ↓               │
│                                    ┌──────────────┐         │
│                                    │   REPORT     │         │
│                                    │   Agent      │         │
│                                    │              │         │
│                                    │ • KPIs       │         │
│                                    │ • Insights   │         │
│                                    └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
GCC_Hack/
├── config/
│   ├── db_config.py           # MS SQL connection settings
│   ├── model_config.py        # ML model hyperparameters
│   └── thresholds.py          # Business rule thresholds
├── agents/
│   ├── sense_agent.py         # Step 1: Data ingestion & features
│   ├── predict_agent.py       # Step 2: ML model training & prediction
│   ├── decide_agent.py        # Step 3: Blocking strategy decisions
│   ├── reserve_agent.py       # Step 4: Execute block reservations
│   ├── monitor_agent.py       # Step 5: Track bookings for rebooking
│   ├── optimize_agent.py      # Step 6: Smart rebooking execution
│   └── report_agent.py        # Step 7: Weekly reporting & KPIs
├── models/
│   ├── demand_model.pkl       # Trained demand spike predictor
│   ├── sellout_model.pkl      # Trained sell-out predictor
│   └── price_model.pkl        # Trained price movement predictor
├── utils/
│   ├── db_utils.py            # Database connection & query helpers
│   ├── feature_engineering.py # Feature creation pipelines
│   └── evaluation.py          # Model evaluation metrics
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
├── reports/                   # Generated HTML reports
├── logs/                      # Pipeline execution logs
├── main.py                    # Pipeline orchestrator (CLI)
├── dashboard.py               # Streamlit interactive dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- MS SQL Server with ODBC Driver 17
- Access to the database with all 12 tables

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify ODBC Driver

```bash
# Windows — typically pre-installed
# Linux
sudo apt-get install unixodbc-dev
# Then install the Microsoft ODBC driver for SQL Server
```

---

## Database Configuration

Edit `config/db_config.py` or set environment variables:

```bash
export DB_SERVER="your_server_name"
export DB_DATABASE="your_database_name"
export DB_USERNAME="your_username"
export DB_PASSWORD="your_password"
export DB_DRIVER="ODBC Driver 17 for SQL Server"
```

### Required Tables (12)

| Table | Rows | Description |
|-------|------|-------------|
| Property_Master | 200 | Hotel property details |
| Supplier_Reliability | 8 | Supplier performance metrics |
| Events_Calendar | 144 | City events and festivals |
| City_Demand_Signals | 7,310 | Daily demand multipliers |
| Property_Daily | 146,200 | Daily property-level metrics |
| Room_Mapping | 3,233 | Cross-supplier room equivalences |
| Rate_Snapshots | 19,200 | Point-in-time supplier rates |
| Confirmed_Bookings | 16,000 | Actual bookings data |
| Demand_Block_Actions | 780 | Blocking action log |
| Rebooking_Evaluations | 1,178 | Rebooking decision log |
| Weekly_Demand_ByCity | 530 | Aggregated weekly demand |
| Weekly_KPI_Summary | 53 | Weekly KPI rollups |

---

## Running the Pipeline

### Full Pipeline (train + execute + report)

```bash
python main.py --week-start 2026-02-10 --train
```

### With Pre-trained Models

```bash
python main.py --week-start 2026-02-10
```

### Report Only

```bash
python main.py --week-start 2026-02-10 --report-only
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--week-start` | Target week start date (Monday) | `2026-02-10` |
| `--train` | Force retrain ML models | `False` |
| `--report-only` | Skip execution, generate report only | `False` |

---

## Streamlit Dashboard

Launch the interactive monitoring dashboard:

```bash
streamlit run dashboard.py
```

### Dashboard Pages

1. **🏠 Overview** — Executive KPI summary with blocking and rebooking metrics
2. **📊 Demand & Predictions** — City-level demand trends, seasonality, events
3. **🏨 Blocking Analysis** — Block actions, reasons, city distribution
4. **🔄 Rebooking Performance** — Risk vs profit analysis, savings distribution
5. **📈 KPI Trends** — Historical weekly KPI trend charts
6. **🏢 Property Drill-Down** — Per-property occupancy, blocking, rebooking history
7. **🤖 Agent Pipeline Status** — System architecture, data summary, model status

---

## Agent Descriptions

### Step 1: SENSE Agent
- Connects to MS SQL and ingests all 12 tables
- Validates data quality (missing values, outliers)
- Engineers 50+ ML features: rolling averages, lags, seasonality, event proximity

### Step 2: PREDICT Agent
- **Demand Model** (XGBoost Regressor): Predicts `city_demand_multiplier` 14 days ahead
- **Sell-out Model** (XGBoost Classifier): Predicts `sold_out_flag` probability
- **Price Model** (XGBoost Regressor): Predicts future `net_rate_inr`
- Cross-validated with 5-fold CV; feature importance analysis included

### Step 3: DECIDE Agent
- Calculates composite blocking score using 5 weighted criteria
- Scores: demand (35%), sell-out (30%), revenue (20%), supplier (10%), price (5%)
- Selects optimal rooms-to-block constrained by inventory

### Step 4: RESERVE Agent
- Checks real-time availability from Rate_Snapshots
- Selects best supplier (preferred + refundable + lowest rate)
- Inserts records into `Demand_Block_Actions` table

### Step 5: MONITOR Agent
- Scans confirmed bookings with future check-in dates
- Searches Rate_Snapshots for cheaper equivalent rooms (via Room_Mapping)
- Verifies cancellation safety windows

### Step 6: OPTIMIZE Agent
- Risk scoring: cancellation (40%), supplier (30%), equivalence (20%), timing (10%)
- Profit scoring: `(savings - penalty) / old_cost * 100`
- Executes rebookings and logs to `Rebooking_Evaluations` table

### Step 7: REPORT Agent
- Generates blocking summary, rebooking performance, missed opportunities
- Produces AI-driven recommendations
- Outputs HTML executive report and updates `Weekly_KPI_Summary`

---

## ML Models

### Performance Targets

| Model | Metric | Target |
|-------|--------|--------|
| Demand Spike | MAE | < 0.15 |
| Sell-out Probability | AUC-ROC | > 0.80 |
| Price Movement | MAPE | < 10% |

### Features Used

- **Temporal**: day_of_week, month, is_weekend, week_of_year
- **Demand**: rolling 7/14/30-day averages, lags, seasonality, event multipliers
- **Property**: star_rating, inventory, popularity_index, occupancy trends
- **Supplier**: failure_rate, cancellation_rate, preferred flag
- **Price**: rate lags, rolling means, volatility

---

## Business Logic

### Blocking Decision Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEMAND_SPIKE_THRESHOLD` | 1.3 | Min demand multiplier |
| `SELLOUT_PROBABILITY_THRESHOLD` | 0.55 | Min sell-out probability |
| `MIN_BLOCKING_SCORE` | 0.50 | Min composite score |
| `MAX_BLOCK_FRACTION` | 0.40 | Max inventory fraction to block |

### Rebooking Decision Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SAVINGS_INR` | ₹200 | Min savings to consider |
| `MIN_NET_PROFIT_INR` | ₹100 | Min net profit after penalty |
| `MIN_EQUIVALENCE_SCORE` | 0.80 | Min room match quality |
| `MAX_RISK_SCORE` | 50 | Max acceptable risk (0-100) |

---

## Success Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Blocking | Sell-out capture rate | > 70% |
| Rebooking | Margin recovery rate | > 5% |
| Models | Demand MAE | < 0.15 |
| Models | Sell-out AUC-ROC | > 0.80 |
| Models | Price MAPE | < 10% |

---

## License

Built for GCC FUSIONX Hackathon — Problem Statement 4.

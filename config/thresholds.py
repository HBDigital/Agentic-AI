"""
Business rule thresholds for blocking and rebooking decisions.
All thresholds are data-driven and should be recalibrated periodically
based on model performance and business outcomes.
"""

# ---------------------------------------------------------------------------
# Demand Blocking Thresholds
# ---------------------------------------------------------------------------

# Minimum predicted demand multiplier to consider blocking
DEMAND_SPIKE_THRESHOLD = 1.3

# Minimum sell-out probability to trigger blocking consideration
SELLOUT_PROBABILITY_THRESHOLD = 0.40

# Minimum revenue score (base_adr_inr * popularity_index) for blocking
REVENUE_SCORE_THRESHOLD = 5000.0

# Maximum acceptable supplier booking failure rate
SUPPLIER_FAILURE_RATE_THRESHOLD = 0.15

# Minimum expected price inflation (%) to justify preemptive blocking
PRICE_INFLATION_THRESHOLD = 0.08

# Minimum composite blocking score to execute a block
MIN_BLOCKING_SCORE = 0.70

# Maximum fraction of inventory that can be blocked per property per night
MAX_BLOCK_FRACTION = 0.40

# Minimum rooms to block (don't block fewer than this)
MIN_ROOMS_TO_BLOCK = 2

# ---------------------------------------------------------------------------
# Blocking Score Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
BLOCKING_WEIGHT_DEMAND = 0.35
BLOCKING_WEIGHT_SELLOUT = 0.30
BLOCKING_WEIGHT_REVENUE = 0.20
BLOCKING_WEIGHT_SUPPLIER = 0.10
BLOCKING_WEIGHT_PRICE = 0.05

# ---------------------------------------------------------------------------
# Rebooking Thresholds
# ---------------------------------------------------------------------------

# Minimum absolute savings (INR) required to consider rebooking
MIN_SAVINGS_INR = 200.0

# Minimum net profit (savings - penalty) in INR
MIN_NET_PROFIT_INR = 100.0

# Minimum room equivalence score to consider rebooking to a different room
MIN_EQUIVALENCE_SCORE = 0.80

# Maximum acceptable risk score (0-100) for rebooking
MAX_RISK_SCORE = 50.0

# Minimum days before cancellation deadline to allow safe rebooking
MIN_DAYS_BEFORE_DEADLINE = 2

# Maximum acceptable cancellation penalty as fraction of old cost
MAX_PENALTY_FRACTION = 0.20

# Minimum supplier reliability (1 - failure_rate) for new supplier
MIN_SUPPLIER_RELIABILITY = 0.85

# Minimum confidence score to execute rebooking
MIN_REBOOKING_CONFIDENCE = 0.60

# ---------------------------------------------------------------------------
# Risk Score Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
RISK_WEIGHT_CANCELLATION = 0.40
RISK_WEIGHT_SUPPLIER = 0.30
RISK_WEIGHT_EQUIVALENCE = 0.20
RISK_WEIGHT_TIMING = 0.10

# ---------------------------------------------------------------------------
# Reporting / KPI Thresholds
# ---------------------------------------------------------------------------

# Minimum margin recovery rate (%) to flag as healthy
MIN_MARGIN_RECOVERY_RATE = 5.0

# Sell-out capture target (%) - percentage of high-risk properties blocked
SELLOUT_CAPTURE_TARGET = 70.0

# Top-N properties to highlight in reports
TOP_N_PROPERTIES = 10

# Weeks of historical data for trend analysis
TREND_LOOKBACK_WEEKS = 8

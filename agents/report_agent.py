"""
REPORT AGENT - Weekly Business Report & KPI Dashboard
=======================================================
Responsibilities:
  1. Generate demand blocking summary (properties blocked, rooms, uplift)
  2. Generate rebooking performance summary (evaluations, margin recovered)
  3. Identify missed opportunities
  4. Produce AI-driven recommendations
  5. Create executive HTML report with visualizations
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import timedelta

from config.thresholds import TOP_N_PROPERTIES, TREND_LOOKBACK_WEEKS, SELLOUT_CAPTURE_TARGET
from utils.db_utils import get_sqlalchemy_engine, read_table, insert_dataframe

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Agent responsible for generating weekly reports,
    KPI summaries, and actionable recommendations.
    """

    def __init__(self, engine=None):
        self.engine = engine or get_sqlalchemy_engine()

    # ------------------------------------------------------------------
    # 1. Blocking Summary
    # ------------------------------------------------------------------

    def generate_blocking_summary(self, week_start: str) -> dict:
        """
        Generate demand blocking KPIs for a given week.

        Returns dict with key metrics and top-N blocked properties.
        """
        logger.info(f"REPORT AGENT: Generating blocking summary for week {week_start}")

        blocks = read_table("demand_block_actions", engine=self.engine)
        blocks["week_start"] = pd.to_datetime(blocks["week_start"])
        week = pd.to_datetime(week_start)

        weekly_blocks = blocks[blocks["week_start"] == week]

        if weekly_blocks.empty:
            return {
                "week_start": week_start,
                "properties_blocked": 0,
                "total_rooms_blocked": 0,
                "expected_uplift_inr": 0,
                "avg_rooms_per_property": 0,
                "top_properties": [],
                "city_distribution": {},
            }

        properties_blocked = weekly_blocks["property_id"].nunique()
        total_rooms = weekly_blocks["rooms_blocked_per_night"].sum()
        expected_uplift = weekly_blocks["expected_revenue_uplift_inr"].sum()

        # Top N properties by uplift
        top = (
            weekly_blocks.nlargest(TOP_N_PROPERTIES, "expected_revenue_uplift_inr")
            [["property_id", "city", "rooms_blocked_per_night",
              "expected_revenue_uplift_inr", "block_reason"]]
            .to_dict("records")
        )

        # City distribution
        city_dist = (
            weekly_blocks.groupby("city")
            .agg(
                blocks=("block_id", "count"),
                rooms=("rooms_blocked_per_night", "sum"),
                uplift=("expected_revenue_uplift_inr", "sum"),
            )
            .to_dict("index")
        )

        return {
            "week_start": week_start,
            "properties_blocked": int(properties_blocked),
            "total_rooms_blocked": int(total_rooms),
            "expected_uplift_inr": round(float(expected_uplift), 2),
            "avg_rooms_per_property": round(total_rooms / max(properties_blocked, 1), 1),
            "top_properties": top,
            "city_distribution": city_dist,
        }

    # ------------------------------------------------------------------
    # 2. Rebooking Summary
    # ------------------------------------------------------------------

    def generate_rebooking_summary(self, week_start: str) -> dict:
        """
        Generate rebooking performance KPIs for a given week.
        """
        logger.info(f"REPORT AGENT: Generating rebooking summary for week {week_start}")

        evals = read_table("rebooking_evaluations", engine=self.engine)
        evals["eval_week"] = pd.to_datetime(evals["eval_week"])
        week = pd.to_datetime(week_start)

        weekly_evals = evals[evals["eval_week"] == week].copy()

        if weekly_evals.empty:
            return {
                "week_start": week_start,
                "total_evaluations": 0,
                "rebook_count": 0,
                "skip_count": 0,
                "total_savings_inr": 0,
                "total_penalties_inr": 0,
                "margin_recovered_inr": 0,
                "avg_profit_per_rebook": 0,
                "rebook_rate_pct": 0,
                "avg_confidence": 0,
                "supplier_distribution": {},
            }

        weekly_evals["decision"] = weekly_evals["decision"].str.strip().str.title()
        rebooked = weekly_evals[weekly_evals["decision"] == "Rebook"]
        skipped = weekly_evals[weekly_evals["decision"] == "Skip"]

        total_savings = rebooked["savings_inr"].sum()
        total_penalty = rebooked["estimated_penalty_inr"].sum()
        margin_recovered = total_savings - total_penalty

        # Supplier performance in rebookings
        sup_dist = {}
        if not rebooked.empty:
            sup_dist = (
                rebooked.groupby("new_supplier")
                .agg(
                    rebookings=("rebooking_eval_id", "count"),
                    avg_savings=("savings_inr", "mean"),
                )
                .to_dict("index")
            )

        return {
            "week_start": week_start,
            "total_evaluations": len(weekly_evals),
            "rebook_count": len(rebooked),
            "skip_count": len(skipped),
            "total_savings_inr": round(float(total_savings), 2),
            "total_penalties_inr": round(float(total_penalty), 2),
            "margin_recovered_inr": round(float(margin_recovered), 2),
            "avg_profit_per_rebook": round(
                margin_recovered / max(len(rebooked), 1), 2
            ),
            "rebook_rate_pct": round(
                len(rebooked) / max(len(weekly_evals), 1) * 100, 1
            ),
            "avg_confidence": round(
                float(rebooked["confidence"].mean()) if not rebooked.empty else 0, 3
            ),
            "supplier_distribution": sup_dist,
        }

    # ------------------------------------------------------------------
    # 3. Missed Opportunities
    # ------------------------------------------------------------------

    def identify_missed_opportunities(self, week_start: str) -> dict:
        """
        Analyze properties that sold out without blocking and
        rebookings that were not executed.
        """
        logger.info(f"REPORT AGENT: Identifying missed opportunities for week {week_start}")

        week = pd.to_datetime(week_start)
        week_end = week + timedelta(days=6)

        # Properties that sold out
        prop_daily = read_table("property_daily", engine=self.engine)
        prop_daily["date"] = pd.to_datetime(prop_daily["date"])

        weekly_soldout = prop_daily[
            (prop_daily["date"] >= week) &
            (prop_daily["date"] <= week_end) &
            (prop_daily["sold_out_flag"] == 1)
        ]

        soldout_props = weekly_soldout["property_id"].unique().tolist()

        # Properties that were blocked
        blocks = read_table("demand_block_actions", engine=self.engine)
        blocks["week_start"] = pd.to_datetime(blocks["week_start"])
        blocked_props = blocks[blocks["week_start"] == week]["property_id"].unique().tolist()

        # Missed = sold out but not blocked
        missed_props = [p for p in soldout_props if p not in blocked_props]

        # Skipped rebookings with high potential
        evals = read_table("rebooking_evaluations", engine=self.engine)
        evals["eval_week"] = pd.to_datetime(evals["eval_week"])
        evals["decision"] = evals["decision"].str.strip().str.title()
        weekly_skipped = evals[
            (evals["eval_week"] == week) &
            (evals["decision"] == "Skip")
        ]

        high_value_skips = weekly_skipped[
            weekly_skipped["savings_inr"] > weekly_skipped["savings_inr"].quantile(0.75)
        ] if not weekly_skipped.empty else pd.DataFrame()

        # Capture rate
        total_soldout = len(soldout_props)
        captured = len([p for p in soldout_props if p in blocked_props])
        capture_rate = round(captured / max(total_soldout, 1) * 100, 1)

        return {
            "week_start": week_start,
            "total_soldout_properties": total_soldout,
            "blocked_and_soldout": captured,
            "missed_soldout_properties": len(missed_props),
            "sellout_capture_rate_pct": capture_rate,
            "capture_target_pct": SELLOUT_CAPTURE_TARGET,
            "capture_gap_pct": round(max(SELLOUT_CAPTURE_TARGET - capture_rate, 0), 1),
            "missed_property_ids": missed_props[:20],
            "high_value_skipped_rebookings": len(high_value_skips),
            "potential_missed_savings_inr": round(
                float(high_value_skips["savings_inr"].sum()) if not high_value_skips.empty else 0, 2
            ),
        }

    # ------------------------------------------------------------------
    # 4. Recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(self, blocking_summary: dict,
                                  rebooking_summary: dict,
                                  missed: dict) -> List[dict]:
        """
        Generate AI-driven recommendations based on weekly performance.
        """
        recommendations = []

        # Blocking recommendations
        if missed.get("capture_gap_pct", 0) > 10:
            recommendations.append({
                "category": "Blocking Strategy",
                "priority": "High",
                "recommendation": (
                    f"Sell-out capture rate ({missed['sellout_capture_rate_pct']}%) "
                    f"is below target ({missed['capture_target_pct']}%). "
                    f"Consider lowering the blocking threshold or expanding coverage "
                    f"to {missed['missed_soldout_properties']} missed properties."
                ),
            })

        if blocking_summary.get("properties_blocked", 0) == 0:
            recommendations.append({
                "category": "Blocking Strategy",
                "priority": "High",
                "recommendation": "No properties were blocked this week. Review demand predictions and thresholds.",
            })

        # Rebooking recommendations
        rebook_rate = rebooking_summary.get("rebook_rate_pct", 0)
        if rebook_rate < 20:
            recommendations.append({
                "category": "Rebooking Optimization",
                "priority": "Medium",
                "recommendation": (
                    f"Low rebooking rate ({rebook_rate}%). "
                    f"Consider relaxing risk thresholds or expanding supplier pool."
                ),
            })

        if missed.get("potential_missed_savings_inr", 0) > 10000:
            recommendations.append({
                "category": "Rebooking Optimization",
                "priority": "High",
                "recommendation": (
                    f"₹{missed['potential_missed_savings_inr']:,.0f} in potential savings "
                    f"were skipped. Review skip reasons and adjust thresholds."
                ),
            })

        # Supplier recommendations
        sup_dist = rebooking_summary.get("supplier_distribution", {})
        if sup_dist:
            best_sup = max(sup_dist.items(), key=lambda x: x[1].get("avg_savings", 0))
            recommendations.append({
                "category": "Supplier Optimization",
                "priority": "Medium",
                "recommendation": (
                    f"Supplier {best_sup[0]} offers the best average savings "
                    f"(₹{best_sup[1].get('avg_savings', 0):,.0f}). "
                    f"Prioritize this supplier for future rebookings."
                ),
            })

        # General
        recommendations.append({
            "category": "General",
            "priority": "Low",
            "recommendation": "Review and recalibrate model predictions against actual outcomes for continuous improvement.",
        })

        return recommendations

    # ------------------------------------------------------------------
    # 5. Weekly KPI Update
    # ------------------------------------------------------------------

    def update_weekly_kpi(self, week_start: str, blocking_summary: dict,
                          rebooking_summary: dict):
        """Insert/update Weekly_KPI_Summary table."""
        kpi_record = {
            "week_start": week_start,
            "properties_blocked": blocking_summary.get("properties_blocked", 0),
            "total_rooms_blocked": blocking_summary.get("total_rooms_blocked", 0),
            "expected_uplift_inr": blocking_summary.get("expected_uplift_inr", 0),
            "rebook_evaluations": rebooking_summary.get("total_evaluations", 0),
            "rebook_count": rebooking_summary.get("rebook_count", 0),
            "margin_recovered_inr": rebooking_summary.get("margin_recovered_inr", 0),
        }

        try:
            kpi_df = pd.DataFrame([kpi_record])
            insert_dataframe(kpi_df, "weekly_kpi_summary", engine=self.engine)
            logger.info(f"Updated Weekly_KPI_Summary for {week_start}")
        except Exception as e:
            logger.error(f"Failed to update Weekly_KPI_Summary: {e}")

    # ------------------------------------------------------------------
    # 6. Executive Report (HTML)
    # ------------------------------------------------------------------

    def create_executive_report(self, week_start: str,
                                output_dir: str = "reports") -> str:
        """
        Generate a comprehensive HTML executive report.

        Returns path to the generated HTML file.
        """
        logger.info("=" * 60)
        logger.info(f"REPORT AGENT: Creating executive report for {week_start}")
        logger.info("=" * 60)

        # Gather all data
        blocking = self.generate_blocking_summary(week_start)
        rebooking = self.generate_rebooking_summary(week_start)
        missed = self.identify_missed_opportunities(week_start)
        recommendations = self.generate_recommendations(blocking, rebooking, missed)

        # Update KPI table
        self.update_weekly_kpi(week_start, blocking, rebooking)

        # Build HTML
        os.makedirs(output_dir, exist_ok=True)
        html = self._render_html_report(week_start, blocking, rebooking, missed, recommendations)

        filepath = os.path.join(output_dir, f"report_{week_start}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Report saved to {filepath}")
        return filepath

    def _render_html_report(self, week_start, blocking, rebooking, missed, recommendations):
        """Render all sections into an HTML document."""
        week_end = (pd.to_datetime(week_start) + timedelta(days=6)).strftime("%Y-%m-%d")

        # Top properties table rows
        top_rows = ""
        for i, p in enumerate(blocking.get("top_properties", []), 1):
            top_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{p.get('property_id','')}</td>
                <td>{p.get('city','')}</td>
                <td>{p.get('rooms_blocked_per_night',0)}</td>
                <td>₹{p.get('expected_revenue_uplift_inr',0):,.0f}</td>
                <td>{p.get('block_reason','')}</td>
            </tr>"""

        # Recommendations rows
        rec_rows = ""
        for r in recommendations:
            priority_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}.get(r["priority"], "#95a5a6")
            rec_rows += f"""
            <tr>
                <td><span style="color:{priority_color};font-weight:bold;">{r['priority']}</span></td>
                <td>{r['category']}</td>
                <td>{r['recommendation']}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Weekly Executive Report — {week_start}</title>
<style>
  :root {{
    --primary: #1a73e8; --success: #27ae60; --danger: #e74c3c;
    --warning: #f39c12; --bg: #f5f7fa; --card: #ffffff;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:#2c3e50; padding:24px; }}
  .header {{ background:linear-gradient(135deg,var(--primary),#0d47a1); color:#fff;
             padding:32px; border-radius:12px; margin-bottom:24px; }}
  .header h1 {{ font-size:28px; margin-bottom:8px; }}
  .header p {{ opacity:0.9; font-size:14px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px; }}
  .kpi-card {{ background:var(--card); border-radius:10px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .kpi-card .label {{ font-size:12px; text-transform:uppercase; color:#7f8c8d; letter-spacing:0.5px; }}
  .kpi-card .value {{ font-size:28px; font-weight:700; margin-top:4px; }}
  .kpi-card .sub {{ font-size:12px; color:#95a5a6; margin-top:2px; }}
  .section {{ background:var(--card); border-radius:10px; padding:24px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
  .section h2 {{ font-size:18px; margin-bottom:16px; color:var(--primary); border-bottom:2px solid var(--primary); padding-bottom:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#f0f4f8; padding:10px 12px; text-align:left; font-weight:600; }}
  td {{ padding:10px 12px; border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#f9fbfd; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; }}
  .badge-success {{ background:#d4edda; color:#155724; }}
  .badge-danger {{ background:#f8d7da; color:#721c24; }}
  .badge-warning {{ background:#fff3cd; color:#856404; }}
  .footer {{ text-align:center; padding:16px; color:#95a5a6; font-size:12px; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Weekly Executive Report</h1>
  <p>Week: {week_start} to {week_end} | Generated by Agentic AI System</p>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="label">Properties Blocked</div>
    <div class="value" style="color:var(--primary)">{blocking.get('properties_blocked',0)}</div>
    <div class="sub">Avg {blocking.get('avg_rooms_per_property',0)} rooms/property</div>
  </div>
  <div class="kpi-card">
    <div class="label">Total Rooms Blocked</div>
    <div class="value">{blocking.get('total_rooms_blocked',0)}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Expected Revenue Uplift</div>
    <div class="value" style="color:var(--success)">₹{blocking.get('expected_uplift_inr',0):,.0f}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Rebookings Executed</div>
    <div class="value" style="color:var(--primary)">{rebooking.get('rebook_count',0)}</div>
    <div class="sub">of {rebooking.get('total_evaluations',0)} evaluated</div>
  </div>
  <div class="kpi-card">
    <div class="label">Margin Recovered</div>
    <div class="value" style="color:var(--success)">₹{rebooking.get('margin_recovered_inr',0):,.0f}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Sellout Capture Rate</div>
    <div class="value" style="color:{'var(--success)' if missed.get('sellout_capture_rate_pct',0) >= 70 else 'var(--danger)'}">{missed.get('sellout_capture_rate_pct',0)}%</div>
    <div class="sub">Target: {missed.get('capture_target_pct',70)}%</div>
  </div>
</div>

<!-- Top Blocked Properties -->
<div class="section">
  <h2>🏨 Top Blocked Properties</h2>
  <table>
    <thead><tr><th>#</th><th>Property</th><th>City</th><th>Rooms/Night</th><th>Expected Uplift</th><th>Reason</th></tr></thead>
    <tbody>{top_rows if top_rows else '<tr><td colspan="6" style="text-align:center;color:#95a5a6;">No blocks this week</td></tr>'}</tbody>
  </table>
</div>

<!-- Rebooking Performance -->
<div class="section">
  <h2>🔄 Rebooking Performance</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Total Evaluations</td><td>{rebooking.get('total_evaluations',0)}</td></tr>
      <tr><td>Rebooked</td><td><span class="badge badge-success">{rebooking.get('rebook_count',0)}</span></td></tr>
      <tr><td>Skipped</td><td><span class="badge badge-warning">{rebooking.get('skip_count',0)}</span></td></tr>
      <tr><td>Total Savings</td><td>₹{rebooking.get('total_savings_inr',0):,.0f}</td></tr>
      <tr><td>Total Penalties</td><td>₹{rebooking.get('total_penalties_inr',0):,.0f}</td></tr>
      <tr><td>Net Margin Recovered</td><td style="font-weight:700;color:var(--success)">₹{rebooking.get('margin_recovered_inr',0):,.0f}</td></tr>
      <tr><td>Avg Profit per Rebook</td><td>₹{rebooking.get('avg_profit_per_rebook',0):,.0f}</td></tr>
      <tr><td>Rebook Rate</td><td>{rebooking.get('rebook_rate_pct',0)}%</td></tr>
      <tr><td>Avg Confidence</td><td>{rebooking.get('avg_confidence',0)}</td></tr>
    </tbody>
  </table>
</div>

<!-- Missed Opportunities -->
<div class="section">
  <h2>⚠️ Missed Opportunities</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Properties Sold Out</td><td>{missed.get('total_soldout_properties',0)}</td></tr>
      <tr><td>Blocked &amp; Sold Out (captured)</td><td>{missed.get('blocked_and_soldout',0)}</td></tr>
      <tr><td>Missed (not blocked)</td><td><span class="badge badge-danger">{missed.get('missed_soldout_properties',0)}</span></td></tr>
      <tr><td>Capture Rate</td><td>{missed.get('sellout_capture_rate_pct',0)}%</td></tr>
      <tr><td>High-Value Skipped Rebookings</td><td>{missed.get('high_value_skipped_rebookings',0)}</td></tr>
      <tr><td>Potential Missed Savings</td><td>₹{missed.get('potential_missed_savings_inr',0):,.0f}</td></tr>
    </tbody>
  </table>
</div>

<!-- Recommendations -->
<div class="section">
  <h2>💡 AI-Driven Recommendations</h2>
  <table>
    <thead><tr><th>Priority</th><th>Category</th><th>Recommendation</th></tr></thead>
    <tbody>{rec_rows}</tbody>
  </table>
</div>

<div class="footer">
  <p>Generated by Agentic AI — Demand Blocking &amp; Smart Rebooking System</p>
</div>

</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # 7. Trend Analysis (historical KPIs)
    # ------------------------------------------------------------------

    def get_kpi_trends(self) -> pd.DataFrame:
        """Load Weekly_KPI_Summary for trend charts."""
        kpi = read_table("weekly_kpi_summary", engine=self.engine)
        kpi["week_start"] = pd.to_datetime(kpi["week_start"])
        return kpi.sort_values("week_start").tail(TREND_LOOKBACK_WEEKS * 7)

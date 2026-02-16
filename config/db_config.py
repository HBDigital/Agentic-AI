"""
Database configuration for MS SQL Server connection.
Update these settings with your actual database credentials.
"""
import os
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# MS SQL Server connection settings
# Override via environment variables or edit the defaults below.
# ---------------------------------------------------------------------------

DB_SERVER = os.getenv("DB_SERVER", "52.172.98.46")
DB_DATABASE = os.getenv("DB_DATABASE", "Demandrebooking")
DB_USERNAME = os.getenv("DB_USERNAME", "hbdusers")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0eman96b00k1ing$")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_PORT = int(os.getenv("DB_PORT", "1433"))

# pyodbc connection string (used as passthrough for SQLAlchemy)
PYODBC_CONNECTION_STRING = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER},{DB_PORT};"
    f"DATABASE={DB_DATABASE};"
    f"UID={DB_USERNAME};"
    f"PWD={DB_PASSWORD};"
    f"TrustServerCertificate=yes;"
)

# SQLAlchemy connection string — uses pyodbc passthrough to handle special chars in password
SQLALCHEMY_CONNECTION_STRING = (
    f"mssql+pyodbc:///?odbc_connect={quote_plus(PYODBC_CONNECTION_STRING)}"
)

# ---------------------------------------------------------------------------
# Table names (matching the schema provided)
# ---------------------------------------------------------------------------
TABLES = {
    "property_master": "Property_Master",
    "supplier_reliability": "Supplier_Reliability",
    "events_calendar": "Events_Calendar",
    "city_demand_signals": "City_Demand_Signals",
    "property_daily": "Property_Daily",
    "room_mapping": "Room_Mapping",
    "rate_snapshots": "Rate_Snapshots",
    "confirmed_bookings": "Confirmed_Bookings",
    "demand_block_actions": "Demand_Block_Actions",
    "rebooking_evaluations": "Rebooking_Evaluations",
    "weekly_demand_bycity": "Weekly_Demand_ByCity",
    "weekly_kpi_summary": "Weekly_KPI_Summary",
}

# ---------------------------------------------------------------------------
# Query settings
# ---------------------------------------------------------------------------
QUERY_TIMEOUT = 120  # seconds
BATCH_SIZE = 5000    # rows per batch for large inserts

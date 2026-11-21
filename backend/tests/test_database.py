"""Database security, SQLGlot parsing, and MySQL execution tests."""
import pytest
from app.services.db import ReadOnlyGateway
from app.exceptions import QuerySecurityError, DatabaseConnectionError


@pytest.fixture(scope="module")
def gateway():
    return ReadOnlyGateway()


def test_sql_validator_permits_safe_select(gateway):
    safe_sql = (
        "SELECT p.productName AS product, ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS value "
        "FROM orders o "
        "JOIN orderdetails od ON o.orderNumber = od.orderNumber "
        "JOIN products p ON od.productCode = p.productCode "
        "WHERE o.status != 'Cancelled' AND o.orderDate >= :target_start AND o.orderDate < :target_end "
        "GROUP BY p.productCode, p.productName "
        "ORDER BY value DESC "
        "LIMIT :limit_n"
    )
    gateway.validate_sql(safe_sql)  # Should not raise


def test_sql_validator_rejects_select_star(gateway):
    with pytest.raises(QuerySecurityError) as exc_info:
        gateway.validate_sql("SELECT * FROM orders")
    assert "SELECT *" in str(exc_info.value)


def test_sql_validator_rejects_mutations(gateway):
    dangerous_queries = [
        "DROP TABLE orders",
        "DELETE FROM orders WHERE orderNumber = 10100",
        "UPDATE orders SET status = 'Shipped'",
        "INSERT INTO orders (orderNumber) VALUES (99999)",
        "ALTER TABLE orders ADD COLUMN test INT",
        "TRUNCATE TABLE orderdetails",
    ]
    for sql in dangerous_queries:
        with pytest.raises(QuerySecurityError):
            gateway.validate_sql(sql)


def test_sql_validator_rejects_comments_and_separators(gateway):
    with pytest.raises(QuerySecurityError):
        gateway.validate_sql("SELECT orderNumber FROM orders; DROP TABLE orders")

    with pytest.raises(QuerySecurityError):
        gateway.validate_sql("SELECT orderNumber FROM orders -- inline comment")

    with pytest.raises(QuerySecurityError):
        gateway.validate_sql("SELECT orderNumber FROM orders /* block comment */")


def test_sql_validator_rejects_unapproved_tables(gateway):
    unapproved_queries = [
        "SELECT user, host FROM mysql.user",
        "SELECT table_name FROM information_schema.tables",
        "SELECT * FROM sys.version",
        "SELECT id FROM unauthorized_retail_table",
    ]
    for sql in unapproved_queries:
        with pytest.raises(QuerySecurityError):
            gateway.validate_sql(sql)


def test_sql_validator_rejects_dangerous_functions(gateway):
    dangerous_funcs = [
        "SELECT SLEEP(5) FROM orders",
        "SELECT BENCHMARK(1000000, MD5(1)) FROM orders",
        "SELECT LOAD_FILE('/etc/passwd') FROM orders",
    ]
    for sql in dangerous_funcs:
        with pytest.raises(QuerySecurityError):
            gateway.validate_sql(sql)


def test_live_mysql_schema_discovery(gateway):
    snapshot = gateway.catalog.get_snapshot()
    assert "orders" in snapshot.tables
    assert "orderdetails" in snapshot.tables
    assert "products" in snapshot.tables
    assert "customers" in snapshot.tables

    # Check date range
    assert snapshot.date_range["min_order_date"].startswith("2003")
    assert snapshot.date_range["max_order_date"].startswith("2005")


def test_live_mysql_query_execution(gateway):
    sql = (
        "SELECT p.productLine AS line, ROUND(SUM(od.quantityOrdered * od.priceEach), 2) AS revenue "
        "FROM orders o "
        "JOIN orderdetails od ON o.orderNumber = od.orderNumber "
        "JOIN products p ON od.productCode = p.productCode "
        "WHERE o.status != 'Cancelled' AND o.orderDate >= :target_start AND o.orderDate < :target_end "
        "GROUP BY p.productLine "
        "ORDER BY revenue DESC"
    )
    params = {"target_start": "2004-01-01", "target_end": "2004-12-31"}
    result = gateway.execute(sql, params)

    assert result["id"] is not None
    assert result["columns"] == ["line", "revenue"]
    assert len(result["rows"]) > 0
    assert result["duration_ms"] >= 0
    assert result["query_hash"] is not None

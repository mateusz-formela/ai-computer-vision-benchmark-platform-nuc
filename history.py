from influxdb import InfluxDBClient

from config import (
    INFLUX_HOST,
    INFLUX_PORT,
    INFLUX_DATABASE,
)

client = InfluxDBClient(
    host=INFLUX_HOST,
    port=INFLUX_PORT,
)

client.switch_database(INFLUX_DATABASE)


def get_history(limit=1000):

    query = f"""
    SELECT *
    FROM yolo_summary
    ORDER BY time DESC
    LIMIT {limit}
    """

    result = client.query(query)

    return list(result.get_points())
def delete_runs(run_ids):

    for run_id in run_ids:

        client.query(
            f"DELETE FROM yolo_metrics WHERE run_id='{run_id}'"
        )

        client.query(
            f"DELETE FROM yolo_summary WHERE run_id='{run_id}'"
        )
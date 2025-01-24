# data_processing/log_parser.py

import re
from datetime import datetime
import pandas as pd

def parse_log_file(log_content):
    """
    Parse log files into a structured format.
    """
    logs = []
    for line in log_content.splitlines():
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.*)', line)
        if match:
            timestamp, level, message = match.groups()
            logs.append({
                "timestamp": datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"),
                "level": level,
                "message": message
            })
    return pd.DataFrame(logs)
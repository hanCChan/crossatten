"""Small dependency-free Markdown tables for stage-owned reports."""
import math


def table(frame, decimals=4):
    columns = list(frame.columns)

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{decimals}f}" if math.isfinite(value) else "NA"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines.extend("| " + " | ".join(format_value(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)

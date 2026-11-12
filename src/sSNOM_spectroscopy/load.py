import pandas as pd


def load_neaspec_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find beginning of actual data
    data_start = next(
        i for i, line in enumerate(lines)
        if line.startswith("Row")
    )

    # Keep metadata as raw lines
    metadata = lines[:data_start]

    metadata = parse_metadata(metadata)

    # Read only the actual CSV table with pandas
    data = pd.read_csv(
        filepath,
        skiprows=data_start,
        sep="\t",
    )

    data.drop(columns=[col for col in data.columns
                       if col.startswith("Unnamed")],
              inplace=True)

    return data, metadata


def parse_metadata(lines):
    metadata = {}

    for line in lines:
        # Remove "# " and whitespace
        line = line.lstrip("# ").strip()

        # Split on tabs and remove empty fields
        parts = [part.strip() for part in line.split("\t") if part.strip()]

        if not parts:
            continue

        # First field is the name
        name = parts[0].rstrip(":")

        # Unit is optional
        if len(parts) > 1 and parts[1].startswith("["):
            unit = parts[1].strip("[]")
            values = parts[2:]
        else:
            unit = None
            values = parts[1:]

        # Store single values as a scalar
        if len(values) == 1:
            values = values[0]

        metadata[name] = {
            "unit": unit,
            "value": values
        }

    return metadata

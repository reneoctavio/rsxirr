#!/usr/bin/env python3
"""Compute IRR goldens with a spreadsheet, so the suite can pin what users see.

People check our answers against Excel. This builds a flat ODS holding every
case as a row of cash flows plus an `=IRR()` formula, hands it to LibreOffice
headless, and reads back what the spreadsheet computed. The result is written
to `goldens.csv`, which is committed — the suite runs without LibreOffice, and
regenerating is a deliberate act with a visible diff.

`goldens.csv` as committed came from Excel, not from LibreOffice: convert
`cases.fods` to xlsx, format the computed column as a number with 15 decimals
so the export carries the value and not the rounded display, and save the
column over the `expected` field.

Running this script overwrites those goldens with LibreOffice's. The two agreed
to 1e-9 on 71 of 72 cases, and disagreed on one — a 480-period annuity where
LibreOffice converges to -198%, below where a rate means anything, and Excel
does not. Use it to add or reshape cases; take the goldens back from Excel
before committing them.

Usage: python3 generate_goldens.py
"""

import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FODS = os.path.join(HERE, "cases.fods")
GOLDENS = os.path.join(HERE, "goldens.csv")
PRODUCTION = os.path.join(HERE, "production_flows.json")

# Hand-written cases: the shapes a solver gets wrong, not the shapes it gets right.
SYNTHETIC = [
    # name, cash flow, guess (None = spreadsheet default)
    ("textbook", [-100.0, 39.0, 59.0, 55.0, 20.0], None),
    ("late_return", [-100.0, 0.0, 0.0, 74.0], None),
    ("negative_rate", [-1000.0, 100.0, 200.0, 300.0], None),
    ("long_annuity", [-172545.848122807] + [787.735232517999] * 480, None),
    ("near_zero", [-100.0, 50.0, 50.5], None),
    ("large_scale", [-1e9, 3e8, 4e8, 5e8], None),
    ("tiny_scale", [-1e-6, 4e-7, 4e-7, 4e-7], None),
    # Sign changes in the middle: more than one root can satisfy the equation,
    # and which one comes back is decided by where the search starts. These are
    # the cases where our answer and someone's spreadsheet can legitimately
    # differ, so they are pinned with an explicit guess on both sides.
    ("two_roots_low_guess", [-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, -1.0], 0.0),
    ("two_roots_high_guess", [-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, -1.0], 0.9),
    ("alternating", [-100.0, 230.0, -132.0], None),
    ("alternating_low_guess", [-100.0, 230.0, -132.0], 0.05),
    ("alternating_high_guess", [-100.0, 230.0, -132.0], 0.5),
]


def column(index):
    """Spreadsheet column name for a zero-based index."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def build_fods(cases):
    rows = []
    for row, (name, flow, guess) in enumerate(cases, start=1):
        cells = "".join(
            f'<table:table-cell office:value-type="float" office:value="{value!r}"/>'
            for value in flow
        )
        last = column(len(flow))  # column A holds the name, so the flow ends here
        args = f"[.B{row}:.{last}{row}]"
        if guess is not None:
            args += f";{guess!r}"
        rows.append(
            f'<table:table-row>'
            f'<table:table-cell office:value-type="string"><text:p>{name}</text:p></table:table-cell>'
            f"{cells}"
            f'<table:table-cell table:formula="of:=IRR({args})" office:value-type="float"/>'
            f"</table:table-row>"
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:of="urn:oasis:names:tc:opendocument:xmlns:of:1.2"
 office:version="1.2" office:mimetype="application/vnd.oasis.opendocument.spreadsheet">
<office:body><office:spreadsheet><table:table table:name="irr">
{"".join(rows)}
</table:table></office:spreadsheet></office:body></office:document>"""


def spreadsheet_values(cases):
    """The last cell of every row, as computed by LibreOffice."""
    with open(FODS, "w") as handle:
        handle.write(build_fods(cases))

    out_csv = os.path.join(HERE, "cases.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "csv", "--outdir", HERE, FODS],
        check=True, capture_output=True, timeout=600,
    )

    computed = []
    with open(out_csv) as handle:
        for row in csv.reader(handle):
            cells = [c for c in row if c != ""]
            computed.append(cells[-1] if cells else "")
    os.remove(out_csv)
    return computed


def parse(cell):
    """A percentage as the spreadsheet renders it, or None where it gave up."""
    cell = cell.strip()
    if not cell.endswith("%"):
        return None  # Err:523 and friends: the spreadsheet did not converge
    return float(cell[:-1]) / 100.0


def main():
    cases = list(SYNTHETIC)

    # Real project cash flows, scaled so the largest is 1. IRR is invariant
    # under a positive scaling, so the golden is unchanged and the amounts are
    # gone — these come from production and carry no figures of anyone's.
    #
    # They are monthly, and a monthly rate sits near 1%. From the spreadsheet's
    # default 10% guess only 17 of these 60 converge inside its iteration
    # limit; from 1% all of them do. Worth knowing beyond this file: a customer
    # checking a monthly IRR with a bare `=IRR(range)` sees an error most of
    # the time, and the number they cannot see is not the one in dispute.
    if os.path.exists(PRODUCTION):
        with open(PRODUCTION) as handle:
            for index, flow in enumerate(json.load(handle)):
                cases.append((f"production_{index:03d}", flow, 0.01))

    computed = spreadsheet_values(cases)
    if len(computed) != len(cases):
        sys.exit(f"expected {len(cases)} rows back, got {len(computed)}")

    with open(GOLDENS, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "guess", "expected", "flow"])
        gave_up = 0
        for (name, flow, guess), cell in zip(cases, computed):
            rate = parse(cell)
            if rate is None:
                gave_up += 1
            writer.writerow([
                name,
                "" if guess is None else repr(guess),
                "" if rate is None else repr(rate),
                " ".join(repr(v) for v in flow),
            ])

    print(f"{len(cases)} cases, {gave_up} the spreadsheet could not solve -> {GOLDENS}")


if __name__ == "__main__":
    main()

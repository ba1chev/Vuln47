from collections import Counter

from source.preprocessing.explorer import Explorer
from source.preprocessing.data_preprocessing.data_loading.data_loader import DataLoader


class DataExplorer(Explorer[str]):
    """Prints EDA over one split: class balance, lengths, top CWEs/projects.

    A code-side companion to Notebook 01 — streams a split through 'DataLoader'
    and reports the imbalance and distribution facts that justify the training
    and evaluation choices. Side-effecting (prints only), no return value.
    """

    def __init__(self):
        self.loader = DataLoader()

    def explore(self, input: str) -> None:
        self.analyze_split(input)
        self.show_examples(input)

    def analyze_split(self, path: str) -> None:
        """Aggregate counts/lengths/CWEs/projects over one full split and print them."""
        n = 0
        n_vuln = 0
        lengths = []
        cwe_counter = Counter()
        project_counter = Counter()
        for rec in self.loader.load(path):
            n += 1
            target = rec.get("target", 0)
            if target == 1:
                n_vuln += 1
                for c in rec.get("cwe") or []:  # cwe only meaningful for vulnerable rows
                    cwe_counter[c] += 1
            lengths.append(len(rec.get("func", "")))
            project_counter[rec.get("project", "?")] += 1

        n_safe = n - n_vuln
        pct_vuln = 100 * n_vuln / n if n else 0
        # median via a sort keeps this dependency-free (no numpy needed here)
        lengths.sort()
        median_len = lengths[len(lengths) // 2] if lengths else 0
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        print(f"\n{'=' * 60}")
        print(f"SPLIT: {path}")
        print(f"{'=' * 60}")
        print(f"  Total functions:  {n:,}")
        print(f"  Vulnerable (1):   {n_vuln:,} ({pct_vuln:.2f}%)")
        print(f"  Safe (0):         {n_safe:,} ({100 - pct_vuln:.2f}%)")
        print(f"  Imbalance:        1 vulnerable per {n_safe / n_vuln:.1f} safe")
        print(f"  Code length (chars): median={median_len:,}  avg={avg_len:,.0f}  max={lengths[-1]:,}")
        print(f"  Top 5 CWE (among vulnerable):")
        for cwe, cnt in cwe_counter.most_common(5):
            print(f"      {cwe}: {cnt}")
        print(f"  Number of projects: {len(project_counter)}  (top 3: {', '.join(p for p, _ in project_counter.most_common(3))})")

    def show_examples(self, path: str) -> None:
        """Print the first vulnerable and first safe function found in the split."""
        vuln_ex, safe_ex = None, None
        for rec in self.loader.load(path):
            if rec.get("target") == 1 and vuln_ex is None:
                vuln_ex = rec
            elif rec.get("target") == 0 and safe_ex is None:
                safe_ex = rec
            if vuln_ex and safe_ex:  # got one of each — stop scanning
                break

        print(f"\n{'=' * 60}")
        print("EXAMPLE VULNERABLE FUNCTION (target=1)")
        print(f"{'=' * 60}")
        print(f"  Project: {vuln_ex.get('project')}  CWE: {vuln_ex.get('cwe')}  CVE: {vuln_ex.get('cve')}")
        print("  --- code (first 500 chars) ---")
        print("  " + vuln_ex["func"][:500].replace("\n", "\n  "))

        print(f"\n{'=' * 60}")
        print("EXAMPLE SAFE FUNCTION (target=0)")
        print(f"{'=' * 60}")
        print(f"  Project: {safe_ex.get('project')}")
        print("  --- code (first 500 chars) ---")
        print("  " + safe_ex["func"][:500].replace("\n", "\n  "))

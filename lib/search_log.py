"""Persistent search log for the classifier selection process.

Records every protein set evaluated during Steps 6a–6d: the step name, the
exact set (as a sorted tuple of protein names — so sets of the same proteins
in different orders are recognized as equal), training error, CV error, and
the action taken. Used for audit, debugging, and branch comparison.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class SearchLogEntry:
    step: str
    proteins: tuple
    train_error: Optional[float]
    cv_error: Optional[float]
    action: str
    extra: dict = field(default_factory=dict)


class SearchLog:
    """Append-only log of evaluated protein sets."""

    def __init__(self):
        self.entries: list[SearchLogEntry] = []
        self._seen: set[tuple] = set()

    def add_entry(self, step, proteins, train_error=None, cv_error=None,
                  action="", **extra):
        """Record one evaluated protein set.

        `proteins` is normalized to a sorted tuple of names so the same set
        under different orderings is stored canonically.
        """
        key = tuple(sorted(proteins))
        entry = SearchLogEntry(
            step=step,
            proteins=key,
            train_error=train_error,
            cv_error=cv_error,
            action=action,
            extra=dict(extra),
        )
        self.entries.append(entry)
        self._seen.add(key)
        return entry

    def has_set(self, proteins):
        """True if this exact protein set has been evaluated before."""
        return tuple(sorted(proteins)) in self._seen

    def get_dataframe(self):
        """Return the log as a pandas DataFrame (for saving / inspection)."""
        rows = []
        for e in self.entries:
            row = {
                "step": e.step,
                "n_proteins": len(e.proteins),
                "proteins": "|".join(e.proteins),
                "train_error": e.train_error,
                "cv_error": e.cv_error,
                "action": e.action,
            }
            row.update(e.extra)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, filepath):
        """Save the log as CSV."""
        self.get_dataframe().to_csv(filepath, index=False)

    @classmethod
    def load(cls, filepath):
        """Load a previously saved log from CSV."""
        df = pd.read_csv(filepath)
        log = cls()
        for _, row in df.iterrows():
            proteins = tuple(row["proteins"].split("|")) if row["proteins"] else tuple()
            extra = {k: row[k] for k in df.columns
                     if k not in {"step", "n_proteins", "proteins",
                                  "train_error", "cv_error", "action"}}
            log.add_entry(
                step=row["step"],
                proteins=proteins,
                train_error=row.get("train_error"),
                cv_error=row.get("cv_error"),
                action=row.get("action", ""),
                **extra,
            )
        return log

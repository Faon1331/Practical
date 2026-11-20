from pathlib import Path

import pandas as pd

from app.config import GROUP_BY_COLUMN, LABEL_COLUMNS, display_name


class VenueStore:
    """Профили заведений: сколько отзывов и какая доля из них с каждой меткой.

    Таблицу строит scripts/build_venues.py.
    """

    def __init__(self, path: Path):
        df = pd.read_csv(path, dtype={"branch_id": str})
        self.df = df.set_index("branch_id")

    def __len__(self) -> int:
        return len(self.df)

    def _short(self, branch_id: str, row, share: float | None = None) -> dict:
        item = {
            "branch_id": branch_id,
            "name": row["name"],
            "kind": row["kind"] if pd.notna(row["kind"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "reviews": int(row["reviews"]),
        }
        if share is not None:
            item["share"] = round(share, 3)
        return item

    def get(self, branch_id: str) -> dict | None:
        if branch_id not in self.df.index:
            return None
        row = self.df.loc[branch_id]
        labels = []
        for col in LABEL_COLUMNS:
            count = int(row[col])
            if count == 0:
                continue
            labels.append({
                "label": display_name(col),
                "group": GROUP_BY_COLUMN[col],
                "count": count,
                "share": round(count / row["reviews"], 3),
            })
        labels.sort(key=lambda x: x["share"], reverse=True)
        return {**self._short(branch_id, row), "labels": labels}

    def top(self, column: str, limit: int = 10, min_reviews: int = 10) -> list[dict]:
        df = self.df[self.df["reviews"] >= min_reviews]
        share = df[column] / df["reviews"]
        order = pd.DataFrame({"share": share, "reviews": df["reviews"]})
        order = order[order["share"] > 0].sort_values(["share", "reviews"], ascending=False)
        return [self._short(bid, self.df.loc[bid], float(s))
                for bid, s in order["share"].head(limit).items()]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        mask = self.df["name"].str.contains(query, case=False, regex=False, na=False)
        found = self.df[mask].sort_values("reviews", ascending=False).head(limit)
        return [self._short(bid, row) for bid, row in found.iterrows()]

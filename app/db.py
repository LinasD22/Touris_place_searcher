import os
import ast
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Integer, Float, String, Text, LargeBinary
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


load_dotenv()

PG_USER = os.getenv("PGUSER")
PG_PASSWORD = os.getenv("PGPASSWORD")
PG_HOST = os.getenv("PGHOST")
PG_PORT = os.getenv("PGPORT")
PG_DBNAME = os.getenv("PGDATABASE")
_engine = create_engine(
    f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
    f"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT','5432')}"
    f"/{os.getenv('PGDATABASE2')}",
    future=True,
    echo=False,
)

# ──────────────────────────────────────────────────────────────────────────────
# helpers ­– dominant-colour feature engineering
# ──────────────────────────────────────────────────────────────────────────────
def _safe_literal_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return None


def _create_6d_feature_vector(colors: list[list[int]], k_for_features: int) -> Optional[list[float]]:
    if (
        not isinstance(colors, list)
        or len(colors) < k_for_features
        or not all(isinstance(c, list) and len(c) == 3 for c in colors)
    ):
        return None

    v = colors[0] + colors[k_for_features - 1]          # first + k-th dominant
    return v if len(v) == 6 else None


def _build_feature_df(all_places: pd.DataFrame, k_for_features: int) -> pd.DataFrame:
    all_places = all_places.copy()
    all_places["dom_colors"] = all_places["dom_colors"].apply(_safe_literal_eval)
    all_places["feat6"] = all_places["dom_colors"].apply(
        lambda c: _create_6d_feature_vector(c, k_for_features)
    )
    all_places = all_places.dropna(subset=["feat6"])
    return all_places.reset_index(drop=True)


def _knn_neighbors(
    df_feat: pd.DataFrame,
    target_id: str,
    k_neighbors: int,
    id_col: str = "id",
) -> list[str]:
    if target_id not in df_feat[id_col].values:
        return []

    X = np.vstack(df_feat["feat6"].to_numpy())
    idx_map = {rid: i for i, rid in enumerate(df_feat[id_col])}

    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X)), metric="euclidean")
    knn.fit(X)

    tgt_idx = idx_map[target_id]
    dists, inds = knn.kneighbors(X[tgt_idx].reshape(1, -1))

    nbr_ids = [
        df_feat.iloc[i][id_col]
        for i in inds.flatten()
        if df_feat.iloc[i][id_col] != target_id
    ][: k_neighbors]

    return nbr_ids



def _fetch(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine.begin() as con:
        return pd.read_sql(text(sql), con, params=params)


def get_cities() -> List[Dict]:
    sql = "SELECT id, name FROM city ORDER BY name"
    return _fetch(sql).to_dict("records")


def get_places(city_id: int, *, order_by_score: bool = True) -> List[Dict]:
    sql = """
        SELECT p.id,
               p.name,
               ps.score,
               p.image
        FROM place p
        LEFT JOIN pagerank_score ps ON ps.place_id = p.id
        WHERE p.city_id = :cid
    """
    if order_by_score:
        sql += " ORDER BY ps.score DESC NULLS LAST, p.name"
    else:
        sql += " ORDER BY p.name"
    return _fetch(sql, {"cid": city_id}).to_dict("records")


def get_place(place_id: str) -> Dict:
    sql = """
        SELECT p.*,
               c.name AS city_name
        FROM place p
        JOIN city c ON c.id = p.city_id
        WHERE p.id = :pid
    """
    return _fetch(sql, {"pid": place_id}).iloc[0].to_dict()


def get_dominant_colors(place_id: str) -> list:
    sql = """
        SELECT ARRAY_AGG(ARRAY[red,green,blue] ORDER BY color_order) AS colors
        FROM place_dominant_colors
        WHERE place_id = :pid
        GROUP BY place_id
    """
    res = _fetch(sql, {"pid": place_id})
    return res.iloc[0]["colors"] if not res.empty else []


def get_similar_structural(place_id: str, *, same_city: bool = True) -> List[Dict]:
    city_clause = (
        "AND p.city_id = (SELECT city_id FROM place WHERE id = :pid)"
        if same_city
        else "AND p.city_id <> (SELECT city_id FROM place WHERE id = :pid)"
    )
    sql = f"""
        SELECT sp.similar_place_id AS id,
               p.name,
               sp.sim_score,
               p.city_id
        FROM similar_places sp
        JOIN place p ON p.id = sp.similar_place_id
        WHERE sp.main_place_id = :pid
          {city_clause}
        ORDER BY sp.sim_score DESC
        LIMIT 10
    """
    return _fetch(sql, {"pid": place_id}).to_dict("records")


def get_similar_image(
    place_id: str,
    *,
    same_city: Optional[bool] = None,
    k_neighbors: int = 6,
    k_for_features: int = 3,
) -> List[Dict]:
    # 1) grab every place’s id, name, city, and dominant-colors
    sql_all = """
        SELECT p.id,
               p.name,
               p.city_id,
               ARRAY_AGG(ARRAY[dc.red, dc.green, dc.blue] ORDER BY dc.color_order) AS dom_colors
        FROM place p
        JOIN place_dominant_colors dc ON dc.place_id = p.id
        GROUP BY p.id
    """
    all_places = _fetch(sql_all)
    if all_places.empty:
        return []

    # 2) build 6D feature vectors
    feat_df = _build_feature_df(all_places, k_for_features)
    if place_id not in set(feat_df["id"]):
        return []
    # grab the city of the target so we can filter later
    orig_city = int(feat_df.loc[feat_df["id"] == place_id, "city_id"].iloc[0])

    # 3) run KNN *globally* (n_neighbors = all other points)
    X = np.vstack(feat_df["feat6"].to_numpy())
    idx_map = {rid: i for i, rid in enumerate(feat_df["id"])}
    knn = NearestNeighbors(n_neighbors=len(X), metric="euclidean")
    knn.fit(X)
    tgt_idx = idx_map[place_id]
    dists, inds = knn.kneighbors(X[tgt_idx].reshape(1, -1))

    # 4) flatten and exclude the target itself
    all_nbr_ids = [
        feat_df.iloc[i]["id"]
        for i in inds.flatten()
        if feat_df.iloc[i]["id"] != place_id
    ]

    # 5) filter by same_city flag
    filtered = []
    for nid in all_nbr_ids:
        city_id = int(feat_df.loc[feat_df["id"] == nid, "city_id"].iloc[0])
        if same_city is True and city_id == orig_city:
            filtered.append(nid)
        elif same_city is False and city_id != orig_city:
            filtered.append(nid)
        elif same_city is None:
            filtered.append(nid)
        if len(filtered) >= k_neighbors:
            break

    if not filtered:
        return []

    # 6) fetch the image bytes + dom_colors for those filtered IDs
    sql_details = """
        SELECT p.id,
               p.name,
               p.city_id,
               p.image,
               ARRAY_AGG(ARRAY[dc.red,dc.green,dc.blue] ORDER BY dc.color_order) AS dominant_colors
        FROM place p
        JOIN place_dominant_colors dc ON dc.place_id = p.id
        WHERE p.id = ANY(:ids)
        GROUP BY p.id
    """
    df_details = _fetch(sql_details, {"ids": filtered})

    # 7) restore original neighbor order
    order_map = {pid: i for i, pid in enumerate(filtered)}
    df_details["order"] = df_details["id"].map(order_map)
    records = []
    for row in df_details.sort_values("order").to_dict("records"):
        row["dominant_colors"] = row.get("dominant_colors") or row.get("dom_colors")
        records.append({
            "id": row["id"],
            "name": row["name"],
            "city_id": row["city_id"],
            "image": row["image"],
            "dominant_colors": row["dominant_colors"],
        })
    return records


def plot_color_palette(dominant_colors_list, image_id_for_title=None, small=False):
    valid_colors_normalized = []
    for color in dominant_colors_list:
        if isinstance(color, list) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            valid_colors_normalized.append([c/255.0 for c in color])
    if not valid_colors_normalized:
        return
    num_colors = len(valid_colors_normalized)
    if small:
        fig_width = max(2.5, num_colors * 0.6)
        fig_height = max(0.5, fig_width / 6)
    else:
        fig_width = max(5, num_colors * 1.2)
        fig_height = max(1.5, fig_width / 5)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    for i, color_norm in enumerate(valid_colors_normalized):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color_norm))
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    title = "Dominant Color Palette"
    if image_id_for_title:
        title += f" for Image ID: {image_id_for_title}"
    ax.set_title(title)
    plt.tight_layout(pad=0.5)
    return fig


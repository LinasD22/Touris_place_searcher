#mpiexec -n 4 python MPI_processing.py

from mpi4py import MPI
import os, json, pickle, numpy as np, pandas as pd, cv2
from collections import Counter
from io import BytesIO
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sqlalchemy import (
    create_engine,
    Integer,
    Float,
    LargeBinary,
    String,
    Text,
    text,
)
import networkx as nx
from dotenv import load_dotenv
from typing import Any, List, Optional, Union

np.seterr(divide="raise", invalid="raise")


def print_row_counts(df: pd.DataFrame, *, include_nulls: bool = False) -> None:
    total_len = len(df)
    for col in df.columns:
        n = total_len if include_nulls else df[col].count()
        print(f"{col}: {n:,}")


def quality_report(dataframe):
    for variable in dataframe.columns:
        null_sum = dataframe[variable].isnull().sum()
        print(null_sum)
        print(len(dataframe[variable]))
        print(variable)
        percent = null_sum / len(dataframe[variable])
        print("")
        print(f"----------------------------------{variable}----------------------------------")
        print(f" null value persentage: {percent*100}, null value count: {null_sum}")
        cardinality = dataframe[variable].nunique()
        print(f" unique value count: {cardinality}")


def drop_na_values(df):
    mask_na = df.isna().any(axis=1)
    df_no_na = df[~mask_na]

    mask_dup = df_no_na.duplicated(subset=["name"], keep="first")
    cleaned_df = df_no_na[~mask_dup]

    removed_ids = pd.concat(
        [df[mask_na]["name"], df_no_na[mask_dup]["name"]]
    ).tolist()

    return cleaned_df, removed_ids


def dedupe_dataframe(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    mask_dupes = df.duplicated(subset=subset, keep=keep)

    n_dupes = mask_dupes.sum()
    if verbose:
        cols = "all columns" if subset is None else ", ".join(subset)
        print(f"🗑  {n_dupes:,} duplicate rows found on {cols} (keeping='{keep}').")

    if inplace:
        df.drop(index=df.index[mask_dupes], inplace=True)
        return df

    return df.loc[~mask_dupes].copy()


def remove_vertices_by_name(G: nx.Graph, names_to_remove: List[str]):
    nodes_to_drop = [
        n for n, d in G.nodes(data=True)
        if n in names_to_remove or d.get("name") in names_to_remove
    ]

    G.remove_nodes_from(nodes_to_drop)
    print(f"🗑 {len(nodes_to_drop)} vertices removed.")
    return G


def from_bytes_to_image(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    img.show()
    print("Image bytes:", len(img_bytes) if img_bytes else "None")


def imread_from_bytes(
    data: Union[bytes, bytearray, memoryview],
    flags: int = cv2.IMREAD_COLOR,
):
    if data is None:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    return img


def get_dominant_colors(img_bytes, k=7):
    img = imread_from_bytes(img_bytes)
    if img is None:
        return [[0, 0, 0]] * k

    h, w = img.shape[:2]
    if max(h, w) > 200:
        s = 200 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))

    pix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    k = min(k, len(pix))
    km = MiniBatchKMeans(k, n_init=1).fit(pix)
    order = np.argsort([-Counter(km.labels_)[i] for i in range(k)])
    cols = km.cluster_centers_[order].astype(int).tolist()

    return (cols + [[0, 0, 0]] * k)[:k]


def k_means_on_chunk(df, k=7):
    rec = []

    for _, r in df.iterrows():
        for o, rgb in enumerate(get_dominant_colors(r["image"], k), 1):
            rec.append(
                {
                    "place_id": r["id"],
                    "color_order": o,
                    "red": rgb[0],
                    "green": rgb[1],
                    "blue": rgb[2],
                }
            )

    return pd.DataFrame(rec)


def id_for_name(name: str, df: pd.DataFrame) -> Optional[Any]:
    mask = df["name"] == name
    if not mask.any():
        print(f' Name "{name}" not found in DataFrame.')
        return None

    if mask.sum() > 1:
        print(f'Multiple rows match "{name}" – returning the first ID.')

    return df.loc[mask, "id"].iloc[0]


def get_similar_places_df(objects_to_props, loc_df):
    object_list = list(objects_to_props.keys())
    n = len(object_list)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            objA = object_list[i]
            objB = object_list[j]

            propsA = objects_to_props[objA]
            propsB = objects_to_props[objB]

            if not isinstance(propsA, set):
                propsA = set(propsA)
            if not isinstance(propsB, set):
                propsB = set(propsB)

            intersection = propsA.intersection(propsB)
            union = propsA.union(propsB)

            id1 = id_for_name(objA, loc_df)
            id2 = id_for_name(objB, loc_df)

            if id1 is None or id2 is None:
                print("Either id1 or id2 are None")
                continue

            overlap_ratio = len(intersection) / len(union) if union else 0.0
            rows.append(
                {
                    "main_place_id": id1,
                    "similar_place_id": id2,
                    "sim_score": overlap_ratio,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["main_place_id", "similar_place_id", "sim_score"],
    )


def copy_df(df: pd.DataFrame, table: str, dtypes: dict[str, Any], eng) -> None:
    df.to_sql(
        table,
        eng,
        if_exists="append",
        index=False,
        dtype=dtypes,
        method="multi",
        chunksize=10_000,
    )


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    load_dotenv()
    env = {
        k: os.getenv(k)
        for k in [
            "PGUSER",
            "PGPASSWORD",
            "PGHOST",
            "PGPORT",
            "PGDATABASE",
            "PGDATABASE2",
        ]
    }

    eng_raw = create_engine(
        f"postgresql://{env['PGUSER']}:{env['PGPASSWORD']}@"
        f"{env['PGHOST']}:{env['PGPORT']}/{env['PGDATABASE']}"
    )

    city_df = pd.read_sql_table("city", eng_raw)
    place_df = pd.read_sql_table("place", eng_raw)
    category_df = pd.read_sql_table("category", eng_raw)
    place_category_df = pd.read_sql_table("place_category", eng_raw)

    place_df, _ = drop_na_values(place_df)
    city_df = dedupe_dataframe(city_df)
    place_df = dedupe_dataframe(place_df)
    category_df = dedupe_dataframe(category_df)
    place_category_df = dedupe_dataframe(place_category_df)

    chunks = np.array_split(place_df, size)
else:
    env = city_df = category_df = place_category_df = None
    chunks = None

env = comm.bcast(env, root=0)              # ← creds to all ranks
local = comm.scatter(chunks, root=0)       # ← each rank gets a slice

# ---------- each rank: colour extraction ---------------------------------
local_dc = k_means_on_chunk(local)

gathered = comm.gather(local_dc, root=0)   # ← gather partial tables

# ---------- rank-0: rest of pipeline -------------------------------------
if rank == 0:
    dom_cols = pd.concat(gathered, ignore_index=True)

    with open("./data/cities_prop.json") as f:
        props = json.load(f)

    similar_places = get_similar_places_df(props, place_df)

    with open("./data/graph.pickle", "rb") as f:
        graph = pickle.load(f)

    pr_df = pd.DataFrame(
        [
            {"place_id": n, "score": s}
            for n, s in nx.pagerank(graph).items()
        ]
    )

    eng_proc = create_engine(
        f"postgresql://{env['PGUSER']}:{env['PGPASSWORD']}@"
        f"{env['PGHOST']}:{env['PGPORT']}/{env['PGDATABASE2']}"
    )

    with eng_proc.begin() as c:
        c.execute(
            text(
                """TRUNCATE place_category,
                          place,
                          category,
                          city,
                          place_dominant_colors,
                          similar_places,
                          pagerank_score
                   RESTART IDENTITY CASCADE"""
            )
        )

    copy_df(city_df, "city", {"id": Integer, "name": String(255)}, eng_proc)

    copy_df(
        category_df,
        "category",
        {"id": String(50), "name": String(255), "description": Text},
        eng_proc,
    )

    copy_df(
        place_df,
        "place",
        {
            "id": String(50),
            "name": String(255),
            "description": Text,
            "location_url": String(1024),
            "image": LargeBinary,
            "city_id": Integer,
        },
        eng_proc,
    )

    copy_df(
        place_category_df,
        "place_category",
        {"place_id": String(50), "category_id": String(50)},
        eng_proc,
    )

    copy_df(
        dom_cols,
        "place_dominant_colors",
        {
            "color_order": Integer,
            "red": Integer,
            "green": Integer,
            "blue": Integer,
            "place_id": String(50),
        },
        eng_proc,
    )

    copy_df(
        similar_places,
        "similar_places",
        {
            "main_place_id": String(50),
            "similar_place_id": String(50),
            "sim_score": Float,
        },
        eng_proc,
    )

    copy_df(
        pr_df,
        "pagerank_score",
        {"place_id": String(50), "score": Float},
        eng_proc,
    )

MPI.Finalize()                              #  shutdown

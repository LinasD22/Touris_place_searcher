import json
import os
import pickle
from collections import Counter, defaultdict
from io import BytesIO
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sqlalchemy import create_engine, Integer, Float, LargeBinary, String, Text, text
from dotenv import load_dotenv
import networkx as nx

np.seterr(divide="raise", invalid="raise")


def print_row_counts(df: pd.DataFrame, *, include_nulls: bool = False) -> None:

    total_len = len(df)
    for col in df.columns:
        n = total_len if include_nulls else df[col].count()
        print(f"{col}: {n:,}")


def quality_report(dataframe):

    for variable in dataframe.columns:
        null_sum=dataframe[variable].isnull().sum()
        print(null_sum )
        print(len(dataframe[variable]))
        print(variable)
        percent=(null_sum)/len(dataframe[variable])
        print("")
        print(f"----------------------------------{variable}----------------------------------")
        print(f" null value persentage: {percent*100}, null value count: {null_sum}")
        cardinality = dataframe[variable].nunique()
        print(f" unique value count: {cardinality}")

def drop_na_values(df):
    # rows knocked out because they contain at least one NaN
    mask_na      = df.isna().any(axis=1)
    df_no_na     = df[~mask_na]

    # rows knocked out as duplicates (same "name", keep first)
    mask_dup     = df_no_na.duplicated(subset=["name"], keep="first")
    cleaned_df   = df_no_na[~mask_dup]

    # collect IDs of everything we dropped
    removed_ids  = pd.concat([df[mask_na]["name"], df_no_na[mask_dup]["name"]]).tolist()

    return cleaned_df, removed_ids

def dedupe_dataframe(df: pd.DataFrame,
                     subset: list[str] | None = None,
                     keep: str = "first",
                     inplace: bool = False,
                     verbose: bool = True) -> pd.DataFrame:

    # 1) locate duplicates
    mask_dupes = df.duplicated(subset=subset, keep=keep)

    # 2) log/report
    n_dupes = mask_dupes.sum()
    if verbose:
        if subset is None:
            cols = "all columns"
        else:
            cols = ", ".join(subset)
        print(f"🗑  {n_dupes:,} duplicate rows found on {cols} "
              f"(keeping='{keep}').")

    # 3) drop & return
    if inplace:
        df.drop(index=df.index[mask_dupes], inplace=True)
        return df
    else:
        return df.loc[~mask_dupes].copy()

def remove_vertices_by_name(G: nx.Graph, names_to_remove: List[str]):
    # ensure we don’t mutate the graph while iterating
    nodes_to_drop = [
        n for n, d in G.nodes(data=True)
        if n in names_to_remove or d.get("name") in names_to_remove
    ]

    G.remove_nodes_from(nodes_to_drop)
    print(f"🗑 {len(nodes_to_drop)} vertices removed.")
    return G


def from_bytes_to_image(img_bytes):
    img = Image.open(BytesIO(img_bytes))
    img.show()          # pops up your OS image viewer
    # img.save("output.jpg")  # …or write to disk

    print("Image bytes:", len(img_bytes) if img_bytes else "None")





def imread_from_bytes(data: Union[bytes, bytearray, memoryview],
                      flags: int = cv2.IMREAD_COLOR):

    if data is None:
        return None

    # Convert the byte buffer to a 1-D uint8 array for OpenCV
    arr = np.frombuffer(data, dtype=np.uint8)

    # Decode — exactly what imread() does internally with a file on disk
    img = cv2.imdecode(arr, flags)

    return img



def get_dominant_colors(image_bytes, k_clusters: int = 7, resize_dim: int = 200):

    image = imread_from_bytes(image_bytes)
    if image is None:
        print(f"Warning: Could not read image {image_bytes}. Returning empty list.")
        return []

    # Resize image for speed
    if resize_dim and resize_dim > 0:
        h, w = image.shape[:2]
        if max(h, w) > resize_dim:
            if h > w:
                new_h = resize_dim
                new_w = int(w * (resize_dim / h))
            else:
                new_w = resize_dim
                new_h = int(h * (resize_dim / w))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    num_pixels = pixels.shape[0]
    if num_pixels == 0:
        # print(f"Warning: Image {image_path} (after potential resize) has no pixels. Returning empty list.")
        return []

    actual_k = k_clusters
    if num_pixels < k_clusters:
        actual_k = num_pixels # Use all pixels as clusters if fewer than k_clusters

    if actual_k == 0:
        return []

    # Use MiniBatchKMeans for faster processing
    kmeans = MiniBatchKMeans(n_clusters=actual_k, random_state=42, n_init=1, batch_size=256*3, max_iter=100)
    kmeans.fit(pixels)

    float_rgb_colors = kmeans.cluster_centers_
    labels = kmeans.labels_ # For MiniBatchKMeans, labels might not be for all points if partial_fit is used,

    labels = kmeans.predict(pixels) # Get labels for all pixels in the (potentially resized) image
    label_counts = Counter(labels)

    cluster_info = []
    for cluster_idx in range(actual_k): # Iterate up to actual_k found clusters
        count = label_counts.get(cluster_idx, 0)
        # Ensure cluster_idx is valid for float_rgb_colors if actual_k was reduced
        if cluster_idx < len(float_rgb_colors):
            color = float_rgb_colors[cluster_idx]
            cluster_info.append({'count': count, 'color': color})

    cluster_info.sort(key=lambda x: x['count'], reverse=True)

    dominant_colors_list = [[int(c_val) for c_val in item['color']] for item in cluster_info]

    # Ensure K colors are returned, padding with black if fewer were found.
    while len(dominant_colors_list) < k_clusters:
        dominant_colors_list.append([0, 0, 0])

    return dominant_colors_list[:k_clusters] # Return exactly K colors

def k_means_on_image(df, k):
    rows = []
    for _, r in df.iterrows():
        img_bytes = r["image"]
        colors = get_dominant_colors(img_bytes, k_clusters=k)

        for order, rgb in enumerate(colors, start=1):
            rows.append({
                "place_id":    r["id"],
                "color_order": order,
                "red":   rgb[0],
                "green": rgb[1],
                "blue":  rgb[2],
            })

    return pd.DataFrame(rows,
                        columns=["place_id", "color_order", "red", "green", "blue"])




def id_for_name(name: str, df: pd.DataFrame) -> Optional[Any]:

    mask = df["name"] == name
    if not mask.any():
        print(f' Name "{name}" not found in DataFrame.')
        return None

    # if multiple matches, warn and take the first one
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

            # Make sure both are sets for easy intersection/union
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
            if len(union) > 0:
                overlap_ratio = len(intersection) / len(union)
                row = {
                    "main_place_id": id1,
                    "similar_place_id": id2,
                    "sim_score": overlap_ratio
                }
                rows.append(row)
            else:
                overlap_ratio = 0.0
                row = {
                    "main_place_id": id1,
                    "similar_place_id": id2,
                    "sim_score": overlap_ratio
                }
                rows.append(row)

    similar_places = pd.DataFrame(rows, columns=["main_place_id", "similar_place_id", "sim_score"])

    return similar_places


def copy_df(df: pd.DataFrame, table: str, dtypes: dict[str, Any], eng) -> None:

    df.to_sql(
        table,
        eng,
        if_exists="append",  # tables are truncated just beforehand
        index=False,
        dtype=dtypes,
        method="multi",
        chunksize=10_000,
    )



if __name__ == "__main__":

    # ____________________ Cleaning ____________________
    # ____________________ Cleaning ____________________

    load_dotenv()

    PG_USER = os.getenv("PGUSER")  # e.g. "postgres"
    PG_PASSWORD = os.getenv("PGPASSWORD")  # e.g. "SuperSecret!"
    PG_HOST = os.getenv("PGHOST", "localhost")
    PG_PORT = os.getenv("PGPORT", "5432")
    PG_DBNAME = os.getenv("PGDATABASE")  # e.g. "Collection_places"
    db_url = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
    print(db_url)

    engine = create_engine(
        f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}",
        echo=False,
        future=True,
    )
    city_df = pd.read_sql_table("city", engine)  # ≈ SELECT * FROM city
    place_df = pd.read_sql_table("place", engine)
    category_df = pd.read_sql_table("category", engine)
    place_category_df = pd.read_sql_table("place_category", engine)

    print_row_counts(place_df)
    quality_report(place_df)
    place_df, names_were_removed = drop_na_values(place_df)
    #quality_report(place_df)

    print("\n\n\n\n\n")
    quality_report(city_df)
    print("\n\n\n\n\n")
    quality_report(category_df)
    print("\n\n\n\n\n")
    quality_report(place_category_df)
    print("\n\n\n\n\n")

    city_df = dedupe_dataframe(city_df)
    place_df = dedupe_dataframe(place_df)
    category_df = dedupe_dataframe(category_df)
    place_category_df = dedupe_dataframe(place_category_df)


    # ____________________ dominant_colors ____________________
    # ____________________ dominant_colors ____________________


    place_dominant_colors = k_means_on_image(place_df, 7)
    place_dominant_colors.to_csv("./data/place_dominant_colors.csv", index=False)


    # ____________________ similar_places ____________________
    # ____________________ similar_places ____________________

    with open("./data/cities_prop.json", "r") as file:
        cities_prop = json.load(file)

    similar_places = get_similar_places_df(cities_prop, place_df)
    similar_places.to_csv("./data/similar_places.csv", index=False)


    # ____________________ Pagerank ____________________
    # ____________________ Pagerank ____________________

    with open(f"./data/graph.pickle", 'rb') as file:
        graph = pickle.load(file)

    cleaned_graph = remove_vertices_by_name(graph, names_were_removed)

    pagerank = nx.pagerank(cleaned_graph)

    rows=[]

    for node, rank in pagerank.items():
        print(f"{node}: {rank:.4f}")
        id = id_for_name(node, place_df)
        row = {
            "place_id": id,
            "score": rank
        }
        rows.append(row)
    pagerank_score = pd.DataFrame(rows, columns=["place_id", "score"])

    pagerank_score.to_csv("./data/pagerank_score.csv", index=False)



    # ____________________ Write to new DB ____________________
    # ____________________ Write to new DB ____________________

    PG_DBNAME2 = os.getenv("PGDATABASE2")  # e.g. "Collection_places"

    db_url2 = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME2}"
    print(db_url2)

    engine2 = create_engine(
        f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME2}",
        echo=False,
        future=True,
    )


    with engine2.begin() as conn:
        conn.execute(
            text(
                """
                TRUNCATE TABLE
                    place_category,
                    place,
                    category,
                    city,
                    place_dominant_colors,
                    similar_places,
                    pagerank_score
                RESTART IDENTITY CASCADE;
                """
            )
        )

    # ── 1. CITY ---------------------------------------------------------------
    copy_df(city_df, "city", {"id": Integer, "name": String(255)}, engine2)

    # ── 2. CATEGORY -----------------------------------------------------------
    copy_df(
        category_df,
        "category",
        {"id": String(50), "name": String(255), "description": Text}, engine2
    )

    # ── 3. PLACE --------------------------------------------------------------
    copy_df(
        place_df,
        "place",
        {
            "id": String(50),
            "name": String(255),
            "description": Text,
            "location_url": String(1024),
            "image": LargeBinary,  # BYTEA
            "city_id": Integer,
        }, engine2
    )

    copy_df(
        place_category_df,
        "place_category",
        {"place_id": String(50), "category_id": String(50)},engine2
    )

    #pagerank_score

    copy_df(
        pagerank_score,
        "pagerank_score",
        {
            "place_id": String(50),
            "score": Float
        }, engine2
    )

    #similar_places

    copy_df(
        similar_places,
        "similar_places",
        {
            "main_place_id": String(50),
            "similar_place_id": String(50),
            "sim_score": Float
        }, engine2
    )

    #place_dominant_colors
    copy_df(
        place_dominant_colors,
        "place_dominant_colors",
        {
            "color_order": Integer,
            "red": Integer,
            "green": Integer,
            "blue": Integer,
            "place_id": String(50)
        }, engine2
    )

    print("All tables wiped and reloaded.")

    print("\n---------------- Done ----------------\n ")
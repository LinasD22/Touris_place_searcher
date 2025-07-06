
# Tourist Place Searcher

**Tourist Place Searcher** is a Streamlit application for browsing points of
interest collected from OpenStreetMap and Wikipedia. Data are stored in
PostgreSQL and enriched with dominant colour information, PageRank scores and
similarity metrics.

## Introduction

The project gathers polygons for several cities (Paris, London, Vilnius and
Chicago) and extracts up to eighty tourist locations per city. Raw records are
cleaned to remove duplicates and nulls before being saved to a processed
`Places_processed` database.

## Dataset

Each place contains the following fields:

- `location_name`
- `location_id`
- `location_description`
- `location_url`
- `location_image`
- `city_id`
- `city_name`
- `categories` and `category_id`

  ![image](https://github.com/user-attachments/assets/df7f191b-63e1-4f06-b4d9-d469c62797c2)


After cleaning about 310 unique entries remain.

## Data Processing

Processing scripts compute dominant colours for images using K‑Means and build
lists of visually similar places with k‑Nearest Neighbours. PageRank is
calculated on a graph of wiki links to rank places by importance. Results are
written back to the database.

## Project Structure

- `app/app.py` – Streamlit web interface
- `app/db.py` – database helpers
- `app/processing_no_MPI.py` / `app/MPI_processing.py` – data preparation
- `app/graph.py` – tools for graph export
- `app/data/` – example processed data

## Installation

Install required Python packages (`streamlit`, `sqlalchemy`, `pandas`,
`scikit-learn`, `networkx`, `Pillow`, `python-dotenv`, …) and ensure PostgreSQL
is available. Create an `.env` file with

```
PGUSER=your_db_user
PGPASSWORD=your_password
PGHOST=localhost
PGPORT=5432
PGDATABASE=your_db_name
PGDATABASE2=your_processed_db_name
```

### Windows

```
git clone https://github.com/LinasD22/Touris_place_searcher
psql -U your_db_user -d your_db_name -f database_dump.sql
streamlit run app/app.py
```

### Linux / macOS

```
git clone https://github.com/LinasD22/Touris_place_searcher
psql -U your_db_user -d your_db_name -f database_dump.sql
streamlit run app/app.py
```

SQL dumps for populating the database are available at
<https://drive.google.com/drive/folders/1W79S9tWe2Jle9QRrddqvCxv11sCRJzRd?usp=sharing>.

## User Interface

Starting the app shows the four cities. **View places** displays locations
ranked by PageRank. **View details** reveals a place description, dominant
colour palette and similar locations by image or category. Navigation buttons
allow returning to the places list or city list.

## Conclusion

This project demonstrates collecting data from public APIs, storing it in
**PostgreSQL and analysing it with K‑Means, KNN and PageRank**. The Streamlit UI
presents the results in an accessible way for exploring tourist destinations.


>>>>>>> main


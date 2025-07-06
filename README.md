# Tourist Place Searcher

This repository contains a small Streamlit application for exploring places of interest stored in a PostgreSQL database.  The app allows selecting a city, browsing its places, and viewing additional information such as similar locations based on image or structural properties.

## Contents

- `app/app.py` – main Streamlit interface.
- `app/db.py` – helper functions for database access and similarity queries.
- `app/processing_no_MPI.py` and `app/MPI_processing.py` – scripts for computing dominant colours, similarity measures and page‑rank scores. The MPI variant can distribute processing across workers.
- `app/graph.py` – tools for building and exporting graphs of similar places.
- `app/data/` – sample processed data (subgraphs, pickle files and more).

## Setup

1. Install Python packages.  Typical requirements include `streamlit`, `sqlalchemy`, `pandas`, `numpy`, `scikit-learn`, `networkx`, `Pillow` and `dotenv`.  Use `pip` to install them:
   ```bash
   pip install streamlit sqlalchemy pandas numpy scikit-learn networkx Pillow python-dotenv
   ```
2. Provide database credentials through environment variables or an `.env` file with the following keys:
   `PGUSER`, `PGPASSWORD`, `PGHOST`, `PGPORT`, `PGDATABASE`, `PGDATABASE2`.

## Running the App

Once dependencies and environment variables are ready, launch the application with:

```bash
streamlit run app/app.py
```

This opens a local web page where you can navigate through the available cities and places.  Selecting a place displays its information, image, dominant colour palette and lists of similar points of interest.

## Data Processing

The processing scripts load raw records, compute dominant colours for images, build similarity tables and PageRank scores, and then write the results to another database.  Run `processing_no_MPI.py` for a single‑process workflow or `MPI_processing.py` with `mpiexec` for distributed execution.


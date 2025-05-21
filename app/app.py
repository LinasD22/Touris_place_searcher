import io
from typing import Dict

import streamlit as st
from PIL import Image
import web_helpers
import db

st.set_page_config(page_title="POI Explorer", layout="wide", page_icon="🏙️")

if hasattr(st, "rerun"):
    _rerun = st.rerun
else:
    _rerun = st.experimental_rerun

if "view" not in st.session_state:
    st.session_state.view = "cities"
    st.session_state.city_id: int | None = None
    st.session_state.place_id: str | None = None

def _image_from_bytes(b: bytes):
    return Image.open(io.BytesIO(b))

def _card_button(label, key=None, help=None):
    st.markdown(
        f"""
        <style>
        .poi-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(60,60,60,0.08);
            margin-bottom: 1rem;
        }}
        .poi-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #263238;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    return st.button(f"🔎 {label}", key=key, help=help)

def _similar_block(title: str, items: list[Dict]):
    st.markdown(f"#### {title}")
    for itm in items:
        with st.container():
            key_prefix = f"n-{title}-{itm['id']}"
            cols = st.columns([2, 3, 2])
            with cols[0]:
                img = itm.get("image")
                if img:
                    st.image(_image_from_bytes(img), caption="Image", use_container_width=True)
                dom_colors = itm.get("dominant_colors")
                if dom_colors:
                    palette_fig = db.plot_color_palette(dom_colors, small=True)
                    st.pyplot(palette_fig)
            with cols[1]:
                st.markdown(f"**{itm['name']}**")
                if "score" in itm:
                    st.markdown(f"**Score:** `{itm['score']:.4f}`")
                st.caption(f"ID: {itm['id']}")
                # Add description/other details if you want
            with cols[2]:
                st.markdown("")
                if st.button("View details", key=key_prefix):
                    st.session_state.place_id = itm["id"]
                    st.session_state.view = "place"
                    _rerun()
            st.divider()

def view_cities():
    st.title("🏙️ Cities")
    st.markdown("Select a city to view its places of interest.")
    for city in db.get_cities():
        with st.container():
            st.markdown(f"### {city['name']}")
            if st.button("View places", key=f"city-{city['id']}", help="View all places in this city"):
                st.session_state.city_id = city["id"]
                st.session_state.view = "places"
                _rerun()
            st.divider()

def view_places():
    st.markdown("___")
    if st.button("← Back to cities"):
        st.session_state.view = "cities"
        _rerun()

    city_id = st.session_state.city_id
    city_name = next(c["name"] for c in db.get_cities() if c["id"] == city_id)
    st.title(f"📍 Places in {city_name}")

    st.markdown("Sort by score of interest. Select a place to see details.")
    places = db.get_places(city_id, order_by_score=True)
    for p in places:
        with st.container():
            cols = st.columns([2, 6, 2])
            with cols[0]:
                if p.get("image"):
                    st.image(_image_from_bytes(p["image"]), use_container_width=True)
            with cols[1]:
                st.markdown(f"#### {p['name']}")
                st.caption(f"Description: {p.get('description', '')[:100]}{'...' if len(p.get('description',''))>100 else ''}")

                if p.get("score") is not None:
                    st.markdown(f"⭐ **Interest Score:** `{p['score']:.4f}`")
            with cols[2]:
                if st.button("View details", key=f"p-{p['id']}"):
                    st.session_state.place_id = p["id"]
                    st.session_state.view = "place"
                    _rerun()
            st.divider()



def view_place():
    p = db.get_place(st.session_state.place_id)
    st.title(p["name"])
    main_cols = st.columns([4, 6])

    with main_cols[0]:
        if p["image"]:
            st.image(_image_from_bytes(p["image"]), use_container_width=True)
        palette_fig = db.plot_color_palette(db.get_dominant_colors(p["id"]), small=False)
        st.pyplot(palette_fig)

    with main_cols[1]:
        st.markdown(f"**City:** {p['city_name']}")
        if p.get("location_url"):
            st.markdown(f"[📍 Location link]({p['location_url']})")
        #st.write(p.get("description", ""))
        new_txt = web_helpers.get_markdown(p.get('description', ''))
        st.markdown(new_txt)
        st.markdown("___")


        _similar_block(
            "🎨 Similar POIs (image, same city)",
            db.get_similar_image(p["id"], same_city=True),
        )
        _similar_block(
            "🌍 Similar POIs (image, other cities)",
            db.get_similar_image(p["id"], same_city=False),
        )

        _similar_block(
            "🗂️ Similar POIs (structural, same city)",
            db.get_similar_structural(p["id"], same_city=True),
        )

    st.markdown("___")
    nav = st.columns([2, 2, 6])
    if nav[0].button("← Back to places"):
        st.session_state.view = "places"
        _rerun()
    if nav[1].button("← Back to cities"):
        st.session_state.view = "cities"
        _rerun()
    nav[2].write("")

pages = {
    "cities": view_cities,
    "places": view_places,
    "place": view_place,
}

pages[st.session_state.view]()

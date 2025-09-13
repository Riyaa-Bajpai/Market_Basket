import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from backend import get_all_items, filtering, recommend_for_basket, mapping, df, get_item_recommendations

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.markdown("<h1 style='color: #4B8BBE;'>🛒 Market Basket Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3>Select items from the list below to get recommendations.</h3>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

# -----------------------------
# LEFT COLUMN: Basket & Recommendations
# -----------------------------
with col_left:
    all_items = get_all_items()  # returns original item names
    user_basket = st.multiselect("Select items for your basket:", options=all_items)

    st.markdown("<h3 style='color: #4CAF50;'>🧺 Your Basket</h3>", unsafe_allow_html=True)
    if user_basket:
        for item in user_basket:
            st.markdown(f"- **{item}**")
    else:
        st.markdown("<i>Your basket is empty.</i>", unsafe_allow_html=True)

    recommendations = []
    if st.button("✨ Get Recommendations"):
        if not user_basket:
            st.warning("Please select at least one item from the basket.")
        else:
            recommendations = get_item_recommendations(user_basket, filtering, df, mapping, top_n=5)
            
            if not recommendations:
                st.info("No recommendations available for the selected items.")
            else:
                st.markdown("<h3 style='color: #FF5722;'>🎯 Recommended Items</h3>", unsafe_allow_html=True)
                for idx, item in enumerate(recommendations, start=1):
                    st.markdown(f"{idx}. **{item}**")
category_colors = {
    "dairy": "#FFD700",       # gold
    "bakery": "#FFA07A",      # light salmon
    "beverages": "#00CED1",   # dark turquoise
    "alcohol": "#8B0000",     # dark red
    "meat": "#FF6347",        # tomato
    "seafood": "#1E90FF",     # dodger blue
    "produce": "#32CD32",     # lime green
    "household": "#DAA520",   # goldenrod
    "personal care": "#FF69B4", # hot pink
    "snacks": "#FF8C00",      # dark orange
    "canned/packaged": "#9370DB", # mediumpurple
    "frozen": "#4682B4",      # steelblue
    "Other": "#A9A9A9"        # dark gray
}
# -----------------------------
# RIGHT COLUMN: Visualizations
# -----------------------------
with col_right:
    if user_basket and recommendations:
        st.markdown("<h3 style='color: #673AB7;'>🔗 Visualizations</h3>", unsafe_allow_html=True)

       # ---- Sankey: Basket → Recommendations ----
        all_nodes = list(set(user_basket + recommendations))
        node_indices = {name: i for i, name in enumerate(all_nodes)}

        # Assign node colors: basket = green, recommendations = orange
        node_colors = []
        for node in all_nodes:
            if node in user_basket:
                node_colors.append("#4CAF50")  # Basket: green
            else:
                node_colors.append("#8B0000")  # Recommendation: orange

        # Build source, target, value, and link colors
        source, target, value, link_colors = [], [], [], []
        for b in user_basket:
            for r in recommendations:
                source.append(node_indices[b])
                target.append(node_indices[r])
                value.append(1)
                # Link color: gradient-like between basket and recommendation
                link_colors.append("lightblue")  # you can pick any color

        # Create Sankey chart
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors
            )
        )])

        fig_sankey.update_layout(title_text="Basket → Recommendations Flow", font_size=12)
        st.plotly_chart(fig_sankey, use_container_width=True)

        # ---- PyVis Network ----
        net = Network(height="400px", width="100%", notebook=False)
        for item in user_basket:
            net.add_node(item, label=item, color="#4CAF50", shape='dot', size=20)
        for item in recommendations:
            net.add_node(item, label=item, color="#4C5BAF", shape='dot', size=20)
        for b in user_basket:
            for r in recommendations:
                net.add_edge(b, r)
        
        net.save_graph("basket_recommendation.html")
        HtmlFile = open("basket_recommendation.html", 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=450)

        # ---- Category Visualizations ----
        df_viz = pd.DataFrame({
            "Item": user_basket + recommendations
        })
        df_viz["Item_clean"] = df_viz["Item"].str.strip().str.lower()
        df_viz["Category"] = df_viz["Item_clean"].apply(lambda x: mapping.get(x, "Other"))
        df_viz["Type"] = ["Basket"] * len(user_basket) + ["Recommendation"] * len(recommendations)

        # Sunburst
        fig_sunburst = px.sunburst(
        df_viz, path=["Type", "Category", "Item"],
        color="Category", 
        color_discrete_map=category_colors,
        title="Basket vs Recommendations by Category")
        fig_sunburst.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_sunburst, use_container_width=True)

    else:
        st.markdown("<h3 style='color: #555;'>📊 Visualizations</h3>", unsafe_allow_html=True)
        st.markdown("<i>Select items from the left to see recommendations and visualizations here.</i>", unsafe_allow_html=True)

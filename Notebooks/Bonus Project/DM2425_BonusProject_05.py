import pandas as pd
import streamlit as st
import plotly.express as px
import umap
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# Use the raw file URL from your GitHub repository
url = 'https://raw.githubusercontent.com/DanieLLL5/Data-Mining-Project-Group5/main/Datasets/dataset_with_clusters.csv'
data = pd.read_csv(url)

# Data Preparation
data['cluster_labels'] = data['cluster_labels'].astype(str)  # Ensure cluster labels are strings

# Streamlit Application Setup
st.title("Cluster Exploration and Visualization")

# Add Logo to the Top Right Corner
st.sidebar.image("logo.png", width=150)

# Tabs for Navigation
with st.sidebar:
    selected_tab = st.radio("Navigation", ["Home", "Cluster Visualization", "Cluster Comparison", "Filters","About Us"])

# Tab 1: Home
if selected_tab == "Home":
    st.header("Dataset Overview")
    st.write(data.head())

# Tab 2: Cluster Visualization
elif selected_tab == "Cluster Visualization":
    st.header("UMAP and t-SNE Visualization")

    @st.cache_data
    def compute_umap(data, features, random_state=42):
        umap_model = umap.UMAP(random_state=random_state)
        return umap_model.fit_transform(data[features])

    @st.cache_data
    def compute_tsne(data, random_state=90):
        tsne_model = TSNE(n_components=2, random_state=random_state)
        return tsne_model.fit_transform(data)

    # Perform UMAP dimensionality reduction
    cuisine_preferences = [col for col in data.columns if "CUI_" in col]
    customer_activity = ["Orders_Weekday", "Orders_Weekend", "Early_Morning", "Morning", "Afternoon", "Evening", "Night","dayswus","recency"]
    features = cuisine_preferences + customer_activity
    umap_result = compute_umap(data, features)

    # Convert UMAP result to a DataFrame
    umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
    umap_df["cluster_labels"] = data["cluster_labels"]

    # Add Cluster Selector
    selected_clusters = st.multiselect(
        "Select Clusters to Display:", 
        options=data["cluster_labels"].unique(),
        default=data["cluster_labels"].unique()
    )
    
    # Filter Data Based on Selection
    filtered_umap_df = umap_df[umap_df["cluster_labels"].isin(selected_clusters)]

    # Plot UMAP visualization
    umap_fig = px.scatter(
        filtered_umap_df, 
        x="UMAP1", 
        y="UMAP2", 
        color="cluster_labels",
        title="UMAP Cluster Visualization",
        labels={"UMAP1": "UMAP Dimension 1", "UMAP2": "UMAP Dimension 2"},
        template="plotly_white"
    )
    st.plotly_chart(umap_fig, use_container_width=True)

    # Perform t-SNE for customer activity
    perspective1_normalized = data[customer_activity]
    tsne_result1 = compute_tsne(perspective1_normalized)

    # Convert t-SNE result to a DataFrame
    tsne_df1 = pd.DataFrame(tsne_result1, columns=["TSNE1", "TSNE2"])
    tsne_df1["customer_activity_labels"] = data["customer_activity_labels"]

    # Interactive Plotly t-SNE Visualization for Customer Activity
    st.subheader("Customer Activity Perspective (t-SNE)")
    tsne_fig1 = px.scatter(
        tsne_df1,
        x="TSNE1",
        y="TSNE2",
        color="customer_activity_labels",
        title="t-SNE Visualization - Customer Activity Perspective",
        labels={"TSNE1": "t-SNE Dimension 1", "TSNE2": "t-SNE Dimension 2"},
        template="plotly_white",
    )
    st.plotly_chart(tsne_fig1, use_container_width=True)

    # Perform t-SNE for cuisine preferences
    perspective2_normalized = data[cuisine_preferences]
    tsne_result2 = compute_tsne(perspective2_normalized)

    # Convert t-SNE result to a DataFrame
    tsne_df2 = pd.DataFrame(tsne_result2, columns=["TSNE1", "TSNE2"])
    tsne_df2["cuisine_preferences_labels"] = data["cuisine_preferences_labels"]

    # Interactive Plotly t-SNE Visualization for Cuisine Preferences
    st.subheader("Cuisine Preferences Perspective (t-SNE)")
    tsne_fig2 = px.scatter(
        tsne_df2,
        x="TSNE1",
        y="TSNE2",
        color="cuisine_preferences_labels",
        title="t-SNE Visualization - Cuisine Preferences Perspective",
        labels={"TSNE1": "t-SNE Dimension 1", "TSNE2": "t-SNE Dimension 2"},
        template="plotly_white",
    )
    st.plotly_chart(tsne_fig2, use_container_width=True)

# Tab 3: Cluster Comparison
elif selected_tab == "Cluster Comparison":
    st.header("Cluster Comparison")

    # Box Plot
    st.subheader("Box Plot by Feature")
    selected_feature = st.selectbox(
        "Select Feature to Compare:",
        options=[col for col in data.columns if col not in ["cluster_labels", "customer_region", "last_promo", "payment_method"]]
    )

    boxplot_fig = px.box(
        data, 
        x="cluster_labels", 
        y=selected_feature, 
        color="cluster_labels",
        title=f"Box Plot of {selected_feature} by Cluster",
        labels={"cluster_labels": "Cluster", selected_feature: selected_feature},
        template="plotly_white"
    )
    st.plotly_chart(boxplot_fig, use_container_width=True)

    # Cluster Centroids
    st.subheader("Cluster Centroids")
    numerical_columns = data.select_dtypes(include=['number']).columns  # Select only numerical columns
    centroids = data.groupby("cluster_labels")[numerical_columns].mean()  # Calculate centroids only for numerical columns
    st.write(centroids)

    def cluster_profiles(df, label_columns, figsize, cmap="tab10", compare_titles=None):
        """
        Generate cluster profiling visualizations for multiple clustering labels.
        """
        if compare_titles is None:
            compare_titles = [""] * len(label_columns)

        fig, axes = plt.subplots(nrows=len(label_columns), 
                                 ncols=2, 
                                 figsize=figsize, 
                                 constrained_layout=True,
                                 squeeze=False)
        for ax, label, title in zip(axes, label_columns, compare_titles):
            # Filtering df
            drop_cols = [i for i in label_columns if i != label]
            dfax = df.drop(drop_cols, axis=1)

            # Getting the cluster centroids and counts
            centroids = dfax.groupby(by=label, as_index=False).mean()
            counts = dfax.groupby(by=label, as_index=False).count().iloc[:, [0, 1]]
            counts.columns = [label, "counts"]

            # Setting Data
            pd.plotting.parallel_coordinates(centroids, 
                                             label, 
                                             color=sns.color_palette(cmap),
                                             ax=ax[0])

            sns.barplot(x=label, 
                        hue=label,
                        y="counts", 
                        data=counts, 
                        ax=ax[1], 
                        palette=sns.color_palette(cmap),
                        dodge=False, legend=False)

            # Setting Layout
            handles, _ = ax[0].get_legend_handles_labels()
            cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
            ax[0].annotate(text=title, xy=(0.95, 1.1), xycoords='axes fraction', fontsize=13, fontweight='heavy') 
            ax[0].axhline(y=0.5, color="black", linestyle="--")
            ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
            ax[0].set_xticklabels(ax[0].get_xticklabels(), 
                                  rotation=40, ha='right')

            ax[0].legend(handles, cluster_labels,
                         loc='center left', bbox_to_anchor=(1, 0.5), title=label)

            ax[1].set_xticks([i for i in range(len(handles))])
            ax[1].set_xticklabels(cluster_labels)
            ax[1].set_xlabel("")
            ax[1].set_ylabel("Absolute Frequency")
            ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)
        
        st.subheader("Cluster Simple Profiling")
        st.pyplot(fig)

    # Profiling Clusters
    numerical_features = [col for col in data.columns if data[col].dtype != 'object']
    cluster_profiles(
        df=data[numerical_features + ['customer_activity_labels', 'cuisine_preferences_labels', 'cluster_labels']], 
        label_columns=['customer_activity_labels', 'cuisine_preferences_labels', 'cluster_labels'], 
        figsize=(28, 13), 
        compare_titles=["Activity clustering", "Cuisine clustering", "Merged clusters"]
    )

# Tab 4: Filters
elif selected_tab == "Filters":
    st.header("Filter Data and Descriptive Statistics")

    # Region Filter
    unique_regions = data['customer_region'].unique().tolist()
    selected_regions = st.multiselect("Select Regions:", options=unique_regions, default=unique_regions)

    # Last Promo Filter
    unique_promos = data['last_promo'].unique().tolist()
    selected_promos = st.multiselect("Select Last Promo:", options=unique_promos, default=unique_promos)

    # Payment Method Filter
    unique_methods = data['payment_method'].unique().tolist()
    selected_methods = st.multiselect("Select Payment Methods:", options=unique_methods, default=unique_methods)

    # Apply Filters
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data['customer_region'].isin(selected_regions)]
    filtered_data = filtered_data[filtered_data['last_promo'].isin(selected_promos)]
    filtered_data = filtered_data[filtered_data['payment_method'].isin(selected_methods)]

    st.write("Filtered Data", filtered_data)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics by Cluster")
    numerical_columns = filtered_data.select_dtypes(include=['number']).columns
    descriptive_stats = filtered_data.groupby('cluster_labels')[numerical_columns].agg(['mean', 'median', 'std']).reset_index()
    st.write(descriptive_stats)

# Tab 5: About Us
elif selected_tab == "About Us":
    st.header("About Us")

    st.write("""
    ### Team Members
    
    🟦 **Daniel Caridade**  
    *Student ID*: 20211588

    🟩 **Gonçalo Peres**  
    *Student ID*: 20211625

    🟧 **Gonçalo Teles**  
    *Student ID*: 20211684

    🟨 **João Venichand**  
    *Student ID*: 20211664
    """)

    st.info("We are a team of passionate data scientists, eager to explore and innovate with cutting-edge technology.")

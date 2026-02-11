from mini03.data import make_customers
import matplotlib.pyplot as plt
from mini03.features import scale_features
from mini03.cluster import fit_kmeans
from mini03.analyze import describe_clusters
from mini03.evaluate import silhouette

df = make_customers()

scaled_df, scaler = scale_features(df)

model, labels = fit_kmeans(scaled_df, k = 3)
# print(labels.value_counts().sort_index())

df_plot = df.copy()
df_plot["cluster"] = labels.values

summary = describe_clusters(df_plot)
ordered_clusters = summary.index.tolist()

segment_names = {
    ordered_clusters[0]: "VIP",
    ordered_clusters[1]: "Regular",
    ordered_clusters[2]: "Low",
}

df_plot["segment"] = df_plot["cluster"].map(segment_names)

plt.figure(figsize=(7, 5))

for seg in ["Low", "Regular", "VIP"]:
    part = df_plot[df_plot["segment"] == seg]
    plt.scatter(
        part["purchases_30d"],
        part["spend_30d"],
        alpha=0.6,
        label=seg,
    )

plt.xlabel("Purchases (30d)")
plt.ylabel("Spend (30d)")
plt.title("Customer Segments (KMeans)")
plt.legend()
path = "artifacts/mini03/segments.png"
plt.tight_layout()
plt.savefig(path, dpi=150)
print("Saved:", path)


# plt.figure(figsize=(6, 5))
# plt.scatter(df_plot["purchases_30d"], df_plot["spend_30d"], c=df_plot["cluster"], alpha=0.6)
# plt.xlabel("Purchases (30d)")
# plt.ylabel("Spend (30d)")
# plt.title("Customers clusters (KMeans)")
# plt.show()

# print(scaled_df.describe())

# print(df.head())
# print(df.describe())

# plt.figure(figsize=(6, 5))
# plt.scatter(df["purchases_30d"], df["spend_30d"], alpha=0.5)
# plt.xlabel("Purchases (30d)")
# plt.ylabel("Spend (30d)")
# plt.title("Customers distribution")
# plt.show()
    

score = silhouette(scaled_df, labels)
print("silhouette:", round(score, 3))


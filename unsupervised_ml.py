
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Prepare customer data using a Python list
# Each customer has [spending, visits]
customers = [
    [2000, 2],
    [2500, 3],
    [1800, 2],
    [2200, 3],
    [15000, 15],
    [18000, 18],
    [16000, 16],
    [14000, 14],
    [5000, 5],
]

# 2. Scale the customer data
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)

# 3. Create the K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# 4. Train the model and assign clusters
clusters = kmeans.fit_predict(scaled_customers)

# 5. Display each customer's cluster
print("Customer Clustering Results")
print("-" * 35)

for i, cluster in enumerate(clusters):
    print(f"Customer {i + 1} belongs to Cluster {cluster}")

# 6. Display the total number of customers in each cluster
print("\nTotal number of customers in each cluster:")
for cluster in range(3):
    total = sum(clusters == cluster)
    print(f"Cluster {cluster}: {total} customers")


import matplotlib.pyplot as plt

# 7. Create a scatter plot of customer groups
spending = [customer[0] for customer in customers]
visits = [customer[1] for customer in customers]

plt.scatter(
    spending,
    visits,
    c=clusters,
    cmap="viridis",
    s=100
)

#8. Display the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    color="red",
    label="Cluster Centers"
)

plt.xlabel("Customer Spending")
plt.ylabel("Number of Visits")
plt.title("Customer Segmentation Using K-Means")
plt.legend()
plt.grid(True)
plt.show()

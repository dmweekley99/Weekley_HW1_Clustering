import random
import matplotlib.pyplot as plt

# Data Generation
def generate_data(num_points, x_range, y_range):
    """
    Generates random 2D data points within the specified ranges.

    Args:
        num_points (int): Number of random points to generate.
        x_range (tuple): Range for x-coordinates (min, max).
        y_range (tuple): Range for y-coordinates (min, max).

    Returns:
        list: A list of tuples representing the generated 2D points.
    """
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]

# K-Means Implementation
def initialize_centroids(data, k):
    """
    Randomly selects k points from the data to serve as initial centroids.

    Args:
        data (list): List of data points.
        k (int): Number of clusters.

    Returns:
        list: A list of k initial centroid points.
    """
    return random.sample(data, k)

def assign_clusters(data, centroids):
    """
    Assigns each data point to the nearest centroid, forming clusters.

    Args:
        data (list): List of data points.
        centroids (list): List of current centroid points.

    Returns:
        list: A list of clusters, where each cluster is a list of assigned points.
    """
    clusters = [[] for _ in centroids]  # Create an empty cluster for each centroid
    for point in data:
        # Calculate the distance from the point to each centroid
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        # Find the index of the closest centroid
        cluster_index = distances.index(min(distances))
        # Assign the point to the closest cluster
        clusters[cluster_index].append(point)
    return clusters

def update_centroids(clusters):
    """
    Updates the centroids by calculating the mean of each cluster.

    Args:
        clusters (list): List of clusters, where each cluster is a list of points.

    Returns:
        list: A list of new centroid points, calculated as the mean of each cluster.
    """
    return [
        (sum(x for x, _ in cluster) / len(cluster), sum(y for _, y in cluster) / len(cluster))
        for cluster in clusters if cluster  # Check to avoid division by zero
    ]

def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple): The first point (x1, y1).
        point2 (tuple): The second point (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def k_means(data, k, max_iterations):
    """
    Main K-means clustering function.

    Args:
        data (list): List of data points.
        k (int): Number of clusters.
        max_iterations (int): Maximum number of iterations to perform.

    Returns:
        tuple: A tuple containing the final clusters and centroids.
    """
    centroids = initialize_centroids(data, k)  # Initialize centroids
    for iteration in range(max_iterations):
        clusters = assign_clusters(data, centroids)  # Assign points to clusters
        new_centroids = update_centroids(clusters)   # Update centroids
        
        # Visualize the clusters at this iteration
        plot_clusters(clusters, centroids, iteration)

        # Check for convergence: if centroids do not change, break the loop
        if new_centroids == centroids:  
            break
        centroids = new_centroids  # Update centroids for the next iteration
    
    # Final visualization with circles around clusters
    final_plot(clusters, centroids)
    return clusters, centroids

# Visualization
def plot_clusters(clusters, centroids, iteration):
    """
    Visualizes the current state of clustering at a given iteration.

    Args:
        clusters (list): List of clusters, where each cluster is a list of points.
        centroids (list): List of current centroid points.
        iteration (int): Current iteration number for the title.
    """
    plt.figure()
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Define a set of colors for clusters
    for index, cluster in enumerate(clusters):
        plt.scatter(*zip(*cluster), color=colors[index % len(colors)], label=f'Cluster {index + 1}')
    plt.scatter(*zip(*centroids), color='black', marker='X', s=85)  # Plot centroids
    plt.title(f'K-Means Clustering - Iteration {iteration + 1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def final_plot(clusters, centroids):
    """
    Final visualization of clusters with circles around them.

    Args:
        clusters (list): List of clusters, where each cluster is a list of points.
        centroids (list): List of final centroid points.
    """
    plt.figure()
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Define colors for clusters
    for index, cluster in enumerate(clusters):
        plt.scatter(*zip(*cluster), color=colors[index % len(colors)], label=f'Cluster {index + 1}')

        # Draw circles around clusters
        if cluster:  # Only draw if the cluster is not empty
            centroid = centroids[index]  # Get the centroid for the current cluster
            # Calculate the maximum distance from the centroid to any point in the cluster
            max_distance = max(euclidean_distance(point, centroid) for point in cluster)
            # Create a circle with the calculated radius around the centroid
            circle = plt.Circle(centroid, max_distance, color=colors[index % len(colors)], fill=False, linestyle='dotted')
            plt.gca().add_artist(circle)  # Add the circle to the plot

    plt.scatter(*zip(*centroids), color='black', marker='X', s=85)  # Plot centroids
    plt.title('K-Means Final Clustering with Circles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')  # Equal aspect ratio used so circles are not distorted
    plt.show()

def main():
    """
    Main function to execute the K-means clustering process.
    It generates data, prompts the user for input, and starts the K-means algorithm.
    """
    option = int(input("Would you like to manually input coordinates or have coordinates randomly generated? (0 for manual, 1 for random) "))

    if option == 0:
        # Manual input of coordinates
        num_points = int(input("How many data points would you like to enter? "))
        data = []
        for i in range(num_points):
            x = float(input(f"Enter x-coordinate for point {i + 1}: "))
            y = float(input(f"Enter y-coordinate for point {i + 1}: "))
            data.append((x, y))  # Append the entered coordinates as a tuple to the data list

    else:
        # Randomly generate data points
        num_points = int(input("How many data points would you like to generate? "))  
        x_range = (0, 500)  # Range for x-coordinates
        y_range = (0, 500)  # Range for y-coordinates
        data = generate_data(num_points, x_range, y_range)  # Generate random data points

    # Allow user input for number of clusters
    k = int(input("Enter the number of clusters (k): "))
    
    k_means(data, k, 100)  # Run K-means with the specified number of clusters and iterations

if __name__ == "__main__":
    main()  # Execute the main function



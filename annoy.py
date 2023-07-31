import random
from annoy import AnnoyIndex

# Generate random embeddings as placeholders (replace with your actual embeddings)
num_documents = 1000
embedding_size = 300
embeddings = [[random.random() for _ in range(embedding_size)] for _ in range(num_documents)]

# Create AnnoyIndex with the appropriate embedding size
annoy_index = AnnoyIndex(embedding_size, metric='euclidean')

# Add embeddings to the AnnoyIndex
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build the index for efficient searching
annoy_index.build(10)  # Use a suitable number of trees

# Save the AnnoyIndex to disk for later retrieval
annoy_index.save('embedding_index.ann')

# Perform a nearest neighbor search
query_embedding = [random.random() for _ in range(embedding_size)]  # Replace with your query embedding
num_results = 5  # Number of nearest neighbors to retrieve
nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, num_results)

# Print the nearest neighbor indices
print(nearest_neighbors)

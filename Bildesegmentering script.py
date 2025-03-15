import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Sett inn filnavnet til bildet du vil segmentere
image_path = 'Bildesegmentering eksempelbilde.PNG'
k = 3  # Antall klynger --> Du kan endre denne verdien for å se effektene det har på bildet

# Les inn bildet
image = io.imread(image_path)

# Forbered dataene for K-Means
rows, cols, channels = image.shape
pixels = image.reshape(rows * cols, channels)

# Utfør K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(pixels)

# Gjenskap det segmenterte bildet
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(rows, cols, channels).astype(np.uint8)

# Vis resultatene
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(image)
ax[0].set_title('Originalbilde')
ax[0].axis('off')

ax[1].imshow(segmented_img)
ax[1].set_title(f'Segmentert bilde (K={k})')
ax[1].axis('off')

# Vis klyngesentrene
cluster_centers_img = np.zeros((50, 50 * k, channels), dtype=np.uint8)
for i, color in enumerate(kmeans.cluster_centers_.astype(np.uint8)):
    cluster_centers_img[:, i*50:(i+1)*50] = color

ax[2].imshow(cluster_centers_img)
ax[2].set_title('Klyngesentere')
ax[2].axis('off')

plt.show()

import numpy as np
import igraph as ig

# Constants representing the different states of pixels
GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# Global variables for the N-links installation
N_Install = False
N_edges = []
N_weights = []

# Function to compute the beta parameter used in the N-link weights
def compute_beta(img):
    """
    Compute the beta parameter used in the N-link weights.

    Parameters:
    - img: Input image

    Returns:
    - beta: Calculated beta value
    """
    h, w, _ = img.shape

    # Compute differences between neighboring pixels
    left_diff = img[:, 1:] - img[:, :-1]
    upleft_diff = img[1:, 1:] - img[:-1, :-1]
    up_diff = img[1:, :] - img[:-1, :]
    upright_diff = img[1:, :-1] - img[:-1, 1:]

    # Sum of squared differences
    beta = (
        np.sum(left_diff ** 2) +
        np.sum(upleft_diff ** 2) +
        np.sum(up_diff ** 2) +
        np.sum(upright_diff ** 2)
    )

    # Compute normalization factor
    norm_factor = (
        4 * h * w - 3 * h - 3 * w + 2
    )

    # Compute beta
    beta = 1 / (2 * beta / norm_factor)
    
    return beta

# Function to convert pixel coordinates to vertex index
def pixel_to_vertex(x, y, h):
    return y + x * h

# Function to convert vertex index back to pixel coordinates
def vertex_to_pixel(idx, h):
    x = idx // h
    y = idx % h
    return x, y

# Function to compute the weight of the N-link between two pixels
def compute_V(img, i, j, ni, nj, beta, gamma=50):
    """
    Compute the N-link weight between pixel (i, j) and its neighbor (ni, nj).

    Parameters:
    - img: Input image
    - i, j: Coordinates of the first pixel
    - ni, nj: Coordinates of the neighbor pixel
    - beta: Precomputed beta value
    - gamma: Scaling factor for the edge weight

    Returns:
    - Weight of the N-link
    """
    diff = img[i, j] - img[ni, nj]
    distance = np.sqrt(2) if (i - ni) != 0 and (j - nj) != 0 else 1
    weight = gamma / distance * np.exp(-beta * np.sum(diff ** 2))
    return weight

# Function to build the graph for the mincut algorithm
def build_graph(img, mask, bgdGMM, fgdGMM, gamma=50, lamda=1e9, connect_diag=True):
    """
    Build the graph for the mincut algorithm.

    Parameters:
    - img: Input image
    - mask: Current mask
    - bgdGMM: Background GMM
    - fgdGMM: Foreground GMM
    - gamma: Scaling factor for the edge weight
    - lamda: Weight for the T-links
    - connect_diag: Whether to connect diagonal neighbors

    Returns:
    - graph: Constructed graph
    """
    global N_Install, N_edges, N_weights

    h, w, _ = img.shape
    num_pixels = h * w
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pixels + 2)  # Including source and sink
    source = num_pixels
    sink = num_pixels + 1

    if not N_Install:
        beta = compute_beta(img)

    edges = []
    weights = []

    # Compute probabilities for foreground and background
    fg_probs = -fgdGMM.score_samples(img.reshape(-1, 3)).reshape(h, w)
    bg_probs = -bgdGMM.score_samples(img.reshape(-1, 3)).reshape(h, w)

    for i in range(h):
        for j in range(w):
            vertex_id = pixel_to_vertex(i, j, w)
            
            # Add T-links
            if mask[i, j] == GC_BGD:
                edges.append((vertex_id, sink))
                weights.append(lamda)
            elif mask[i, j] == GC_FGD:
                edges.append((vertex_id, source))
                weights.append(lamda)
            else:
                edges.append((vertex_id, source))
                weights.append(bg_probs[i, j])
                edges.append((vertex_id, sink))
                weights.append(fg_probs[i, j])

            if not N_Install:
                # Add N-links
                directions = [(0, 1), (1, 0)]
                if connect_diag:
                    directions.extend([(1, 1), (1, -1)])
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor_vertex_id = pixel_to_vertex(ni, nj, w)
                        weight = compute_V(img, i, j, ni, nj, beta, gamma)
                        N_edges.append((vertex_id, neighbor_vertex_id))
                        N_weights.append(weight)
    
    if not N_Install:
        N_Install = True
    
    # Add edges and weights to the graph
    graph.add_edges(edges + N_edges)
    graph.es['capacity'] = weights + N_weights

    return graph

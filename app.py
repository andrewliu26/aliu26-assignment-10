from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    send_file,
)
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms
import open_clip
from werkzeug.utils import secure_filename
from tqdm import tqdm


def debug_print(msg):
    """Helper function for consistent debug output"""
    print(f"[DEBUG] {msg}")


app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
IMAGE_DIR = "coco_images_resized"
EMBEDDINGS_FILE = "image_embeddings.pickle"
PCA_EMBEDDINGS_FILE = "pca_clip_embeddings.pickle"
MODEL_NAME = "ViT-B/32"
PRETRAINED = "openai"

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables for lazy loading
clip_model = None
preprocess = None
clip_tokenizer = None
clip_embeddings = None
pca = None
pca_embeddings = None
pca_image_names = None

# PCA Cache
PCA_CACHE = {
    "embeddings": None,
    "image_names": None,
    "pca_model": None,
    "n_components": None,
}


def init_clip():
    """Initialize CLIP model only when needed"""
    global clip_model, preprocess, clip_tokenizer
    debug_print("Initializing CLIP model...")
    if clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        debug_print(f"Using device: {device}")
        try:
            clip_model, preprocess, _ = create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED
            )
            clip_model = clip_model.to(device)
            clip_model.eval()
            clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            debug_print("CLIP model initialized successfully")
        except Exception as e:
            debug_print(f"Error initializing CLIP model: {str(e)}")
            raise


def load_clip_embeddings():
    """Load CLIP embeddings only when needed"""
    global clip_embeddings
    debug_print("Loading CLIP embeddings...")
    if clip_embeddings is None:
        try:
            clip_embeddings = pd.read_pickle(EMBEDDINGS_FILE)
            debug_print(f"Loaded {len(clip_embeddings)} CLIP embeddings")
        except Exception as e:
            debug_print(f"Error loading CLIP embeddings: {str(e)}")
            raise


def save_pca_embeddings(embeddings, image_names):
    """Save PCA embeddings to file"""
    debug_print("Saving PCA embeddings to file...")
    data = {"embeddings": embeddings, "image_names": image_names}
    pd.to_pickle(data, PCA_EMBEDDINGS_FILE)
    debug_print("PCA embeddings saved successfully")


def load_cached_pca_embeddings():
    """Load PCA embeddings from file"""
    debug_print("Loading cached PCA embeddings...")
    try:
        data = pd.read_pickle(PCA_EMBEDDINGS_FILE)
        debug_print("PCA embeddings loaded successfully")
        return data["embeddings"], data["image_names"]
    except FileNotFoundError:
        debug_print("No cached PCA embeddings found")
        return None, None


def load_and_preprocess_images(image_dir, max_images=None):
    """Load images and get CLIP embeddings before PCA with file caching"""
    debug_print(f"Loading images from {image_dir} with max_images={max_images}")

    # Try to load from cache file first
    if max_images == 2000:
        cached_embeddings, cached_names = load_cached_pca_embeddings()
        if cached_embeddings is not None:
            debug_print("Using cached PCA embeddings")
            return cached_embeddings, cached_names

    # Initialize CLIP if needed
    init_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    debug_print("Loading and processing images...")
    embeddings = []
    image_names = []

    # Get list of jpg files
    jpg_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(".jpg")]
    if max_images:
        jpg_files = jpg_files[:max_images]

    # Process images in batches for speed
    batch_size = 32
    for i in tqdm(
        range(0, len(jpg_files), batch_size), desc=f"Getting CLIP embeddings"
    ):
        batch_files = jpg_files[i : i + batch_size]
        batch_images = []

        for filename in batch_files:
            img_path = os.path.join(image_dir, filename)
            try:
                image = preprocess(Image.open(img_path))
                batch_images.append(image)
                image_names.append(filename)
            except Exception as e:
                debug_print(f"Error processing {filename}: {e}")
                continue

        if batch_images:
            # Process batch
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_embeddings = clip_model.encode_image(batch_tensor)
                batch_embeddings = F.normalize(batch_embeddings, dim=-1).cpu().numpy()
            embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)

    # Save to cache file if this is the training set
    if max_images == 2000:
        save_pca_embeddings(embeddings, image_names)

    debug_print(f"Successfully processed {len(embeddings)} images")
    return embeddings, image_names


def init_pca(n_components=50):
    """Initialize PCA only when needed"""
    global pca, pca_embeddings, pca_image_names

    # Check if we can use cached PCA model with same components
    if PCA_CACHE["pca_model"] is not None and PCA_CACHE["n_components"] == n_components:
        debug_print("Using cached PCA model")
        pca = PCA_CACHE["pca_model"]
        pca_embeddings = PCA_CACHE["embeddings"]
        pca_image_names = PCA_CACHE["image_names"]
        return

    debug_print(f"Initializing PCA with {n_components} components...")
    try:
        # Load training embeddings
        images, image_names = load_and_preprocess_images(IMAGE_DIR, max_images=2000)
        debug_print(f"Loaded {len(images)} embeddings for PCA training")

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(images)
        debug_print("PCA model fitted successfully")

        # Transform embeddings
        pca_embeddings = pca.transform(images)
        pca_image_names = image_names
        debug_print("PCA embeddings computed successfully")

        # Cache the PCA model and results
        PCA_CACHE["pca_model"] = pca
        PCA_CACHE["embeddings"] = pca_embeddings
        PCA_CACHE["image_names"] = pca_image_names
        PCA_CACHE["n_components"] = n_components

    except Exception as e:
        debug_print(f"Error in PCA initialization: {str(e)}")
        raise


def process_pca_search(image_path, top_k=5):
    """Process PCA-based image search using CLIP features"""
    debug_print(f"Processing PCA image search for {image_path}")
    try:
        # Get CLIP embedding for query image
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image)
            embedding = F.normalize(embedding, dim=-1).cpu().numpy()

        # Apply PCA transform
        query_embedding = pca.transform(embedding)
        similarities = cosine_similarity(query_embedding, pca_embeddings)
        top_indices = np.argsort(-similarities.flatten())[:top_k]

        debug_print("PCA search completed successfully")
        return [pca_image_names[i] for i in top_indices], similarities.flatten()[
            top_indices
        ]
    except Exception as e:
        debug_print(f"Error in PCA search: {str(e)}")
        raise


def process_clip_search(image_path, top_k=5):
    """Process CLIP-based image search"""
    debug_print(f"Processing CLIP image search for {image_path}")
    init_clip()
    load_clip_embeddings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            query_embedding = clip_model.encode_image(image)
            query_embedding = F.normalize(query_embedding, dim=-1).cpu().numpy()

        clip_embeddings_array = np.vstack(clip_embeddings["embedding"].values)
        similarities = cosine_similarity(query_embedding, clip_embeddings_array)
        top_indices = np.argsort(-similarities.flatten())[:top_k]

        debug_print("CLIP image search completed successfully")
        return [
            clip_embeddings.iloc[i]["file_name"] for i in top_indices
        ], similarities.flatten()[top_indices]
    except Exception as e:
        debug_print(f"Error in CLIP image search: {str(e)}")
        raise


def process_clip_text_search(text_query, top_k=5):
    """Process CLIP-based text search"""
    debug_print(f"Processing text search: '{text_query}' with top_k={top_k}")
    init_clip()
    load_clip_embeddings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        debug_print("Tokenizing text...")
        tokens = clip_tokenizer(text_query).to(device)

        debug_print("Generating text embedding...")
        with torch.no_grad():
            query_embedding = clip_model.encode_text(tokens)
            query_embedding = F.normalize(query_embedding, dim=-1).cpu().numpy()

        debug_print("Computing similarities...")
        clip_embeddings_array = np.vstack(clip_embeddings["embedding"].values)
        similarities = cosine_similarity(query_embedding, clip_embeddings_array)
        top_indices = np.argsort(-similarities.flatten())[:top_k]

        results = [clip_embeddings.iloc[i]["file_name"] for i in top_indices]
        scores = similarities.flatten()[top_indices]
        debug_print(f"Found {len(results)} results")
        return results, scores
    except Exception as e:
        debug_print(f"Error in text search processing: {str(e)}")
        raise


def process_hybrid_search(image_path, text_query, weight=0.8, top_k=5):
    """Process hybrid (image + text) search"""
    debug_print(
        f"Processing hybrid search: image={image_path}, text='{text_query}', weight={weight}"
    )
    init_clip()
    load_clip_embeddings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Get image embedding
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = clip_model.encode_image(image)
            image_embedding = F.normalize(image_embedding, dim=-1)

        # Get text embedding
        tokens = clip_tokenizer(text_query).to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)

        # Combine embeddings
        query_embedding = weight * text_embedding + (1 - weight) * image_embedding
        query_embedding = F.normalize(query_embedding, dim=-1).cpu().numpy()

        clip_embeddings_array = np.vstack(clip_embeddings["embedding"].values)
        similarities = cosine_similarity(query_embedding, clip_embeddings_array)
        top_indices = np.argsort(-similarities.flatten())[:top_k]

        debug_print("Hybrid search completed successfully")
        return [
            clip_embeddings.iloc[i]["file_name"] for i in top_indices
        ], similarities.flatten()[top_indices]
    except Exception as e:
        debug_print(f"Error in hybrid search: {str(e)}")
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve images from the image directory"""
    debug_print(f"Serving image: {filename}")
    try:
        return send_from_directory(IMAGE_DIR, filename)
    except FileNotFoundError:
        debug_print(f"Image not found: {filename}")
        return jsonify({"error": "Image not found"}), 404


@app.route("/search", methods=["POST"])
def search():
    try:
        debug_print("Received search request")
        debug_print(f"Form data: {request.form}")
        debug_print(f"Files: {request.files}")

        query_type = request.form["query_type"]
        use_pca = request.form.get("use_pca") == "true"
        n_components = int(request.form.get("num_components", 50))
        top_k = int(request.form.get("top_k", 5))

        debug_print(f"Query type: {query_type}")
        debug_print(f"Use PCA: {use_pca}")
        debug_print(f"Number of components: {n_components}")
        debug_print(f"Top K: {top_k}")

        if use_pca:
            debug_print("Initializing PCA...")
            init_pca(n_components)

        if query_type == "text":
            text_query = request.form.get("text_query", "")
            debug_print(f"Processing text query: {text_query}")
            if not text_query:
                debug_print("Error: No text query provided")
                return jsonify({"error": "No text query provided"}), 400

            results, scores = process_clip_text_search(text_query, top_k)

        elif query_type == "image":
            if "image" not in request.files:
                debug_print("Error: No image file uploaded")
                return jsonify({"error": "No image file uploaded"}), 400
            file = request.files["image"]
            if file.filename == "":
                debug_print("Error: No image file selected")
                return jsonify({"error": "No image file selected"}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            debug_print(f"Saved uploaded image to: {filepath}")

            try:
                if use_pca:
                    results, scores = process_pca_search(filepath, top_k)
                else:
                    results, scores = process_clip_search(filepath, top_k)
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    debug_print(f"Cleaned up temporary file: {filepath}")

        elif query_type == "hybrid":
            text_query = request.form.get("text_query", "")
            if not text_query:
                debug_print("Error: No text query provided for hybrid search")
                return (
                    jsonify({"error": "No text query provided for hybrid search"}),
                    400,
                )

            if "image" not in request.files:
                debug_print("Error: No image file uploaded for hybrid search")
                return (
                    jsonify({"error": "No image file uploaded for hybrid search"}),
                    400,
                )

            file = request.files["image"]
            if file.filename == "":
                debug_print("Error: No image file selected for hybrid search")
                return (
                    jsonify({"error": "No image file selected for hybrid search"}),
                    400,
                )

            weight = float(request.form.get("weight", 0.8))
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            debug_print(f"Saved uploaded image for hybrid search to: {filepath}")

            try:
                results, scores = process_hybrid_search(
                    filepath, text_query, weight, top_k
                )
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    debug_print(f"Cleaned up temporary file: {filepath}")

        else:
            debug_print(f"Error: Invalid query type: {query_type}")
            return jsonify({"error": "Invalid query type"}), 400

        debug_print("Search completed successfully")
        return jsonify(
            {
                "results": [
                    {"filename": name, "score": float(score)}
                    for name, score in zip(results, scores)
                ]
            }
        )

    except Exception as e:
        debug_print(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_print("Starting Flask application...")
    app.run(debug=True, host="0.0.0.0", port=3000)

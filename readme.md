# 🎨 Synthetic Face Generation & PCA Magic 🪄

Welcome to the **Dummy Face Dataset** project! Here, we’re diving into a fun experiment of generating fake face images, extracting powerful features with a simulated **Contourlet Transform**, and then squeezing down the data with some PCA magic!

---

## ⚡️ Project Highlights

- **Dummy Face Generator** 🎭 – Creates unique "faces" for multiple individuals.
- **Simulated Contourlet Transform** 🔍 – Think of it like breaking down an image into multiple frequency bands (simulated, but cool nonetheless)!
- **Feature Extraction & PCA Compression** 📉 – We pull out essential features, then let PCA reduce dimensionality, keeping 95% of the original flavor.
- **Visualization Playground** 🎨 – See example faces, feature vectors, and their PCA-compressed forms side by side.

---

## 🔧 Setup & Dependencies

Before diving in, let’s set up our environment. We’ll be working with:
- `numpy` 🧮 for data handling
- `matplotlib` 🖼️ for visualizations
- `scikit-learn` 🧠 for PCA magic

Install everything with:
```bash
pip install numpy matplotlib scikit-learn
```

---

## 🔍 Code Walkthrough

### 1. **Dataset Generation** – `generate_dummy_faces`
We’re creating a batch of dummy "face images" by generating random patterns. Each individual has their own **base pattern** with slight variations added for uniqueness.

### 2. **Simulated Contourlet Transform** – `simulate_contourlet_transform`
We mimic the Contourlet Transform by creating scaled-down sub-bands from each image, simulating different frequency levels. Perfect for feature extraction!

### 3. **Feature Extraction** – `extract_features`
Flatten each simulated sub-band and glue them together to form a **feature vector** for every face image. This vector holds the "soul" of each face. 😉

### 4. **PCA Compression** – Keep It Tight!
With PCA, we shrink our feature vectors while preserving 95% of the original data’s essence. Fewer dimensions, same vibes!

---

## 🚀 Running the Code

Simply run the main file to see it in action:
```bash
python main.py
```

What you'll see:
1. **Synthetic Face Dataset** 🧑‍🤖 generated with multiple individuals and variations per person.
2. **Feature Extraction** 📊 where we flatten and concatenate the sub-band info.
3. **PCA Dimensionality Reduction** 🌌 with an impressive feature compression.
4. **Visuals** 🎥 showing you:
   - A sample generated face.
   - Its feature vector and the PCA-compressed version.
   - The difference in dimension – you’ll see how much smaller the compressed version is!

---

## 📊 Example Output

Running the code gives you:
- Dimensions of the **generated face dataset** and **feature matrix**.
- **PCA-reduced** feature matrix size.
- Visualizations of an example face, raw feature vector, and reduced features.
- Key **dataset stats** like the number of individuals, images per person, and how much we shrank the data.

---

## 🔍 Explore Further

This project is a *toy example* aimed at showing you how feature extraction and dimensionality reduction works! Real-world applications would use true face images and an actual Contourlet Transform.

--- 

### Let’s Get Generative – Happy Coding! 🚀
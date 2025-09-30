## Project Overview

This project aims to leverage geospatial (GIS) and climatic data to build a machine learning model that predicts the depth of the water table (groundwater level) across selected countries in the Sahel region of West Africa. Accurate water table depth predictions are crucial for sustainable water resource management, especially in regions dependent on rainfed agriculture.

### Objectives

- **Training Phase:**  
  The model is trained using one-time measurements of water table depth, along with a rich set of environmental features, for observation points located in the following countries:
  - **Burkina Faso**
  - **Chad**
  - **Mali**
  - **Mauritania**
  - **Niger**
  - **Senegal**

  The training data includes:
  - In-situ groundwater measurements.
  - Climatic variables (e.g., precipitation, temperature, NDVI).
  - Topographic and hydrological features derived from Digital Elevation Models (DEMs) and GIS analysis.

- **Prediction/Application Phase:**  
  The trained model is used to predict water table depth at pre-defined locations classified as rainfed croplands in these countries:
  - **Benin**
  - **Burkina Faso**
  - **Guinea**
  - **Mali**
  - **Niger**
  - **Togo**

  The focus on rainfed croplands addresses areas where groundwater accessibility directly impacts agricultural productivity and local communities.

### Why This Matters

Providing reliable groundwater depth estimates over large, data-sparse areas empowers governments, NGOs, and local communities to make informed decisions about water resource management, drilling new wells, and planning agricultural activities.

### Acknowledgements

This project was developed by Les solutions géostack, Inc. as part of a research initiative for The World Bank Group.  
For inquiries, contact: info@geostack.ca

---

**Repository:** https://github.com/geo-stack/sahel  
**License:** MIT

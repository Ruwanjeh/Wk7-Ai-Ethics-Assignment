Ai ethics assignment

## Overview
This repository contains solutions for the Wk7 AI Ethics Assignment, covering theoretical analysis, practical AI/IoT/Edge/Quantum tasks, and critical reflection on ethical challenges and future applications.

## Structure
- **Part 1: Theoretical Analysis**
  - Essay answers and case study critique (see below)
- **Part 2: Practical Implementation**
  - Edge AI Prototype: `edge_ai_image_classification.py`
  - AI-Driven IoT Concept: `smart_agriculture_proposal.md`, `smart_agriculture_diagram.mmd`, `smart_agriculture_diagram.png`
  - Ethics in Personalized Medicine: `ethics_personalized_medicine.md`
- **Part 3: Futuristic Proposal**
  - AI for 2030: `futuristic_proposal.md`
- **Bonus Task: Quantum Computing Simulation**
  - Quantum circuit code: `quantum_circuit.py`

---

## Part 1: Theoretical Analysis
Essay answers and case study critique are provided in the report (see your written submission or copy from this repository if needed).

---

## Part 2: Practical Implementation

### Edge AI Prototype
- **File:** `edge_ai_image_classification.py`
- **Description:** Trains a lightweight image classifier (CIFAR-10, 3 classes), converts to TensorFlow Lite, and demonstrates inference for Edge AI.
- **How to Run:**
  1. Install requirements: `pip install tensorflow numpy matplotlib`
  2. Run: `python edge_ai_image_classification.py`
  3. The script will output accuracy and sample predictions.

### AI-Driven IoT Concept

# Smart Agriculture System: AI-Driven IoT Concept

## Overview
This proposal outlines a smart agriculture system that leverages IoT sensors and AI to optimize crop yield prediction and farm management. By integrating real-time sensor data with predictive analytics, farmers can make data-driven decisions to improve productivity and sustainability.

## Sensors Needed
- **Soil Moisture Sensor:** Monitors soil water content
- **Temperature Sensor:** Measures ambient and soil temperature
- **Humidity Sensor:** Tracks air humidity
- **Light Sensor:** Measures sunlight exposure
- **pH Sensor:** Monitors soil acidity/alkalinity
- **Nutrient Sensor:** Detects key soil nutrients (N, P, K)
- **Rain Gauge:** Measures rainfall
- **CO₂ Sensor:** Monitors carbon dioxide levels
- **Camera (optional):** For visual crop health assessment

## AI Model
A regression-based AI model (e.g., Random Forest, Gradient Boosting, or Neural Network) predicts crop yield using sensor data as input features. The model can be trained on historical and real-time data, optionally incorporating weather forecasts and satellite imagery for improved accuracy.

- **Inputs:** Sensor readings (moisture, temperature, humidity, light, pH, nutrients, etc.)
- **Output:** Predicted crop yield (e.g., kg/hectare)

## Workflow
1. **Sensors** collect real-time data from the field.
2. **Data Aggregator** (gateway device) collects and transmits data to a cloud or edge server.
3. **AI Model** processes the data and predicts crop yield.
4. **Dashboard/App** displays actionable insights to farmers for irrigation, fertilization, and harvesting decisions.

See the attached data flow diagram for a visual overview.

## Benefits
- Optimized resource usage (water, fertilizer)
- Increased crop yield and quality
- Early detection of issues (drought, disease)
- Data-driven decision making

## Challenges
- Sensor calibration and maintenance
- Data privacy and security
- Integration with legacy equipment
- Model generalization across different crops and regions

### Ethics in Personalized Medicine

# Ethics in Personalized Medicine: Bias and Fairness

The use of AI in personalized medicine, particularly with datasets like The Cancer Genome Atlas (TCGA), holds great promise for improving patient outcomes by tailoring treatments to individual genetic profiles. However, significant ethical challenges arise, especially regarding bias and fairness.

A primary concern is the underrepresentation of certain ethnic and demographic groups in genomic datasets. If AI models are trained predominantly on data from populations of European descent, their recommendations may be less accurate or even harmful for patients from other backgrounds. This can lead to disparities in diagnosis, treatment efficacy, and overall health outcomes, perpetuating existing inequities in healthcare.

Sources of bias include:
- Sampling bias: Over- or under-representation of specific groups in the training data.
- Measurement bias: Differences in how data is collected or labeled across populations.
- Algorithmic bias: Model architectures or training procedures that inadvertently favor majority groups.

Fairness strategies to address these issues include:
- Diverse and representative training data: Proactively include samples from a wide range of ethnicities, ages, and socioeconomic backgrounds in datasets like TCGA.
- Bias detection and mitigation: Regularly audit AI models for disparate performance across subgroups and apply techniques such as reweighting, resampling, or adversarial debiasing.
- Transparency and explainability: Ensure that AI recommendations are interpretable, allowing clinicians to understand and question the basis for treatment suggestions.
- Stakeholder engagement: Involve patients, clinicians, and ethicists in the design and deployment of AI systems to ensure that diverse perspectives are considered.

In summary, while AI has the potential to revolutionize personalized medicine, careful attention to data diversity, model fairness, and transparency is essential to avoid reinforcing health disparities and to ensure equitable care for all patients.

---

## Part 3: Futuristic Proposal

# AI-Enabled Urban Climate Resilience Platform (2030)

## Problem Statement
By 2030, rapid urbanization and climate change will expose cities to more frequent and severe weather events—heatwaves, floods, and air pollution. Traditional infrastructure and manual response systems are insufficient for real-time adaptation and risk mitigation. There is a critical need for intelligent, adaptive systems that can help cities anticipate, respond to, and recover from climate-related challenges.

## AI Workflow
**Data Inputs:**
- Real-time sensor data (temperature, humidity, air quality, water levels)
- Satellite imagery and weather forecasts
- Social media and citizen reports
- Historical climate and infrastructure data

**Model Type:**
- Multimodal deep learning models for pattern recognition and prediction
- Reinforcement learning for adaptive response strategies
- Graph neural networks for modeling urban infrastructure and dependencies

**Workflow:**
1. Data Collection: IoT sensors and external data sources feed real-time and historical data into the platform.
2. Prediction: AI models forecast extreme weather events, pollution spikes, and infrastructure vulnerabilities.
3. Decision Support: The system recommends adaptive actions (e.g., dynamic traffic rerouting, targeted cooling, floodgate activation) to city managers and emergency responders.
4. Citizen Engagement: Personalized alerts and guidance are sent to residents via mobile apps and public displays.

## Societal Risks and Benefits
**Benefits:**
- Enhanced public safety and reduced disaster impact
- Optimized resource allocation and infrastructure resilience
- Empowered citizens with timely, actionable information
- Data-driven urban planning for long-term sustainability

**Risks:**
- Privacy concerns from pervasive data collection
- Potential for algorithmic bias in resource distribution
- Over-reliance on automated systems, reducing human oversight
- Digital divide: unequal access to alerts and services

**Conclusion:**
An AI-enabled urban climate resilience platform can transform how cities adapt to climate change, saving lives and resources. However, careful governance, transparency, and inclusivity are essential to ensure equitable and ethical deployment.

---

## Bonus Task: Quantum Computing Simulation
- **File:** `quantum_circuit.py`
- **Description:** Simple Qiskit quantum circuit and explanation of its use in AI optimization (e.g., drug discovery).
- **How to Run:**
  1. Install Qiskit: `pip install qiskit`
  2. Run: `python quantum_circuit.py`

---

## Credits & References
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- The Cancer Genome Atlas (TCGA): https://www.cancer.gov/tcga
- TensorFlow, Qiskit, Mermaid CLI
- PLP Academy, IBM Quantum Experience

---

## File List
- `edge_ai_image_classification.py`
- `smart_agriculture_proposal.md`
- `smart_agriculture_diagram.mmd`
- `smart_agriculture_diagram.png`
- `ethics_personalized_medicine.md`
- `futuristic_proposal.md`
- `quantum_circuit.py`

---

For any questions, please refer to the assignment PDF or contact the instructor.

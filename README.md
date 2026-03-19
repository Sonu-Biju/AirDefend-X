# AirDefend-X: Hybrid ADS-B Security Gateway 🛡️✈️

**AirDefend-X** is a multi-layered security framework designed to detect and mitigate **False Data Injection (FDI)** attacks in Automated Dependent Surveillance-Broadcast (ADS-B) telemetry. This project combines cryptographic authentication with machine learning-based behavioral analysis.

## 🌟 Key Features

* **Identity Layer (RSA-2048)**: Implements digital signatures to verify aircraft origin and prevent unauthorized "Ghost" injections.
* **Kinematic Layer (Heuristic Temporal Filter)**: Enforces physical aviation laws (Velocity/Altitude bounds) to catch instantaneous spoofing spikes.
* **Behavioral Layer (Isolation Forest)**: An unsupervised AI model that isolates statistical anomalies in flight patterns (Vertical Rate vs. Ground Speed).
* **Interactive Dashboard**: Real-time ATC visualization built with **Streamlit** and **PyDeck**.

## 📂 Project Structure

* `app.py`: The Streamlit frontend for real-time visualization.
* `logic.py`: The core security engine (RSA + Isolation Forest + Physics Checks).
* `evaluator.py`: Benchmarking script to calculate Precision, Recall, and F1-Score.
* `/data`: Contains `opensky.csv` for simulation and training.

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/V4l7r1x/AirDefend-X.git](https://github.com/V4l7r1x/AirDefend-X.git)
    cd AirDefend-X
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv311
    # Windows:
    .\venv311\Scripts\activate
    # Linux/Mac:
    source venv311/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```

## 📊 Performance & Benchmarking

The system was evaluated using the **OpenSky Network** dataset.

* **Recall**: >98% (Highly effective at catching high-velocity injections).
* **Precision**: 99% (Low false-alarm rate for legitimate commercial traffic).
* **Inference Latency**: <15ms (Suitable for real-time deployment).

---
*Developed as a Cybersecurity Engineering Research Project.*
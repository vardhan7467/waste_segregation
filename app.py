import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Smart Waste Segregation & Recycling System",
    page_icon="♻️",
    layout="wide"
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f4f7fa;
}

.header {
    text-align: center;
    padding: 30px;
}

.header h1 {
    font-size: 42px;
    color: #1b4332;
}

.header p {
    font-size: 18px;
    color: #555;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #2d6a4f;
    margin-top: 15px;
}

.stButton>button {
    background-color: #2d6a4f;
    color: white;
    height: 3em;
    border-radius: 6px;
    font-size: 16px;
    width: 100%;
}

.stButton>button:hover {
    background-color: #1b4332;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>♻️ Smart Waste Segregation & Recycling System</h1>
<p>AI-Based Waste Classification for Sustainable Smart Cities</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="waste_classification_mobilenet.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class_names = [
    "Battery", "Biological", "Cardboard", "Clothes", "Glass",
    "Metal", "Paper", "Plastic", "Shoes", "Trash"
]


waste_info = {
    "Battery": {
        "type": "Hazardous Waste",
        "bin": "🔴 Red Hazardous Waste Bin",
        "recycle": [
            "Collected separately to prevent toxic contamination.",
            "Dismantled in certified e-waste recycling facilities.",
            "Valuable metals like lithium and cobalt are extracted.",
            "Recovered materials are reused in new battery production."
        ],
        "impact": [
            "Prevents heavy metal leakage into soil and groundwater.",
            "Reduces mining of rare earth metals.",
            "Minimizes environmental and human health risks."
        ]
    },
    "Plastic": {
        "type": "Dry Recyclable Waste",
        "bin": "🔵 Blue Recycling Bin",
        "recycle": [
            "Sorted by polymer type (PET, HDPE, etc.).",
            "Washed to remove contamination.",
            "Shredded and melted into pellets.",
            "Reused to manufacture new plastic products."
        ],
        "impact": [
            "Saves approximately 5,774 kWh of energy per ton.",
            "Reduces oil consumption.",
            "Prevents ocean plastic pollution."
        ]
    },
    "Paper": {
        "type": "Dry Recyclable Waste",
        "bin": "🔵 Blue Recycling Bin",
        "recycle": [
            "Collected and sorted by grade.",
            "Pulped and cleaned.",
            "De-inked and processed.",
            "Pressed into new paper sheets."
        ],
        "impact": [
            "Saves 17 trees per ton recycled.",
            "Saves water and energy.",
            "Reduces landfill methane emissions."
        ]
    },
    "Glass": {
        "type": "Dry Recyclable Waste",
        "bin": "🔵 Blue Recycling Bin",
        "recycle": [
            "Sorted by color.",
            "Crushed into cullet.",
            "Melted at high temperature.",
            "Molded into new glass products."
        ],
        "impact": [
            "Reduces raw material extraction.",
            "Saves up to 30% energy.",
            "Can be recycled infinitely."
        ]
    },
    "Metal": {
        "type": "Dry Recyclable Waste",
        "bin": "🔵 Blue Recycling Bin",
        "recycle": [
            "Separated using magnets.",
            "Shredded and melted.",
            "Purified and reused in manufacturing."
        ],
        "impact": [
            "Saves up to 95% energy (aluminum).",
            "Reduces mining impact.",
            "Lowers greenhouse emissions."
        ]
    },
    "Biological": {
        "type": "Organic Waste",
        "bin": "🟢 Green Compost Bin",
        "recycle": [
            "Processed through composting.",
            "Converted into nutrient-rich manure.",
            "Used in agriculture."
        ],
        "impact": [
            "Reduces methane emissions.",
            "Improves soil fertility.",
            "Supports circular economy."
        ]
    },
    "Cardboard": {
        "type": "Dry Recyclable Waste",
        "bin": "🔵 Blue Recycling Bin",
        "recycle": [
            "Shredded into fibers.",
            "Converted into pulp.",
            "Pressed into new cardboard sheets."
        ],
        "impact": [
            "Saves trees.",
            "Reduces deforestation.",
            "Lowers energy consumption."
        ]
    },
    "Clothes": {
        "type": "Textile Waste",
        "bin": "🟡 Textile Collection Bin",
        "recycle": [
            "Reusable clothes donated.",
            "Damaged textiles shredded into fibers.",
            "Used for insulation or industrial materials."
        ],
        "impact": [
            "Reduces landfill overflow.",
            "Promotes sustainable fashion.",
            "Saves water in fabric production."
        ]
    },
    "Shoes": {
        "type": "Non-Biodegradable Waste",
        "bin": "🟡 Special Collection Bin",
        "recycle": [
            "Reusable pairs donated.",
            "Rubber soles recycled.",
            "Materials reused in industrial applications."
        ],
        "impact": [
            "Reduces synthetic waste accumulation.",
            "Encourages reuse culture."
        ]
    },
    "Trash": {
        "type": "Non-Recyclable Waste",
        "bin": "⚫ Black General Waste Bin",
        "recycle": [
            "Sent to sanitary landfills.",
            "Some processed in waste-to-energy plants."
        ],
        "impact": [
            "Proper disposal prevents disease spread.",
            "Reducing trash generation is essential."
        ]
    }
}


def predict_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    index = int(np.argmax(output))
    confidence = float(output[index])

    if index >= len(class_names):
        return "Unknown", 0.0

    return class_names[index], confidence


uploaded_file = st.file_uploader("Upload Waste Image for AI Analysis", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Waste Image", use_column_width=True)

    with col2:
        if st.button("Analyze Waste"):
            label, confidence = predict_image(image)

            if label == "Unknown":
                st.error("Model output mismatch. Please verify model classes.")
            else:
                info = waste_info[label]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"## 🧠 Detected Waste: {label}")
                st.write(f"**Confidence Score:** {confidence:.2%}")
                st.write(f"**Waste Category:** {info['type']}")
                st.write(f"**Recommended Disposal Bin:** {info['bin']}")

                st.markdown("### ♻️ Recycling Process")
                for point in info['recycle']:
                    st.markdown(f"- {point}")

                st.markdown("### 🌍 Environmental Impact")
                for point in info['impact']:
                    st.markdown(f"- {point}")

                st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
---
Smart Waste Segregation & Recycling System | AI-Driven Sustainable Waste Management 🌱
""")


# streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false

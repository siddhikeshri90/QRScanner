"""
app.py

AI-Based QR Code Scam Detection

Pipeline:
1. Upload image or capture using webcam.
2. Detect and decode QR code.
3. Crop QR region and correct perspective.
4. Analyze payload using Branch B.
5. Analyze cropped QR using CNN (Branch A).
6. Fuse both results.
7. Display Low / Medium / High risk verdict.

Run:
    python -m streamlit run app.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

from pyzbar.pyzbar import decode as zbar_decode


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(
    os.path.join(BASE_DIR, "branch_a_image")
)

sys.path.append(
    os.path.join(BASE_DIR, "branch_b_payload")
)

sys.path.append(
    os.path.join(BASE_DIR, "fusion")
)


from payload_pipeline import analyze_payload
from fusion import fuse


# --------------------------------------------------
# TENSORFLOW AND CNN
# --------------------------------------------------

try:

    import tensorflow as tf

    from gradcam import predict_with_gradcam

    HAVE_TF = True

except ImportError:

    HAVE_TF = False


CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "branch_a_image",
    "qr_cnn_model.keras"
)


# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="QR Scam Detector",
    page_icon="🔍",
    layout="centered"
)


st.title("🔍 AI-Based QR Code Scam Detection")

st.caption(
    "Combines CNN-based image tampering detection "
    "with payload risk analysis."
)


# --------------------------------------------------
# LOAD CNN MODEL
# --------------------------------------------------

@st.cache_resource
def load_cnn_model():

    if HAVE_TF and os.path.exists(CNN_MODEL_PATH):

        try:

            model = tf.keras.models.load_model(
                CNN_MODEL_PATH
            )

            return model

        except Exception as e:

            st.warning(
                f"CNN model could not be loaded: {e}"
            )

            return None

    return None


# --------------------------------------------------
# ORDER QR CORNER POINTS
# --------------------------------------------------

def order_points(points):

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    s = points.sum(axis=1)

    diff = np.diff(
        points,
        axis=1
    ).flatten()

    rect[0] = points[np.argmin(s)]       # Top-left
    rect[1] = points[np.argmin(diff)]    # Top-right
    rect[2] = points[np.argmax(s)]       # Bottom-right
    rect[3] = points[np.argmax(diff)]    # Bottom-left

    return rect


# --------------------------------------------------
# QR DETECTION AND PERSPECTIVE CORRECTION
# --------------------------------------------------

def rectify_qr(image, qr):

    try:

        points = np.array(
            [[p.x, p.y] for p in qr.polygon],
            dtype="float32"
        )

        if len(points) < 4:

            raise ValueError(
                "Insufficient QR corner points"
            )

        hull = cv2.convexHull(points)

        perimeter = cv2.arcLength(
            hull,
            True
        )

        approx = cv2.approxPolyDP(
            hull,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            corners = approx.reshape(
                4,
                2
            )

        else:

            corners = cv2.boxPoints(
                cv2.minAreaRect(hull)
            )

        rect = order_points(
            corners
        )

        tl, tr, br, bl = rect

        width = int(
            max(
                np.linalg.norm(br - bl),
                np.linalg.norm(tr - tl)
            )
        )

        height = int(
            max(
                np.linalg.norm(tr - br),
                np.linalg.norm(tl - bl)
            )
        )

        if width < 10 or height < 10:

            raise ValueError(
                "Invalid QR dimensions"
            )

        margin = max(
            10,
            int(max(width, height) * 0.08)
        )

        output_width = width + 2 * margin

        output_height = height + 2 * margin

        destination = np.array(
            [
                [margin, margin],
                [
                    output_width - margin - 1,
                    margin
                ],
                [
                    output_width - margin - 1,
                    output_height - margin - 1
                ],
                [
                    margin,
                    output_height - margin - 1
                ]
            ],
            dtype="float32"
        )

        matrix = cv2.getPerspectiveTransform(
            rect,
            destination
        )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (
                output_width,
                output_height
            ),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return warped

    except Exception:

        # Fallback: use detected QR bounding rectangle

        rect = qr.rect

        x = max(
            0,
            rect.left
        )

        y = max(
            0,
            rect.top
        )

        x2 = min(
            image.shape[1],
            rect.left + rect.width
        )

        y2 = min(
            image.shape[0],
            rect.top + rect.height
        )

        if x2 <= x or y2 <= y:

            return None

        crop = image[
            y:y2,
            x:x2
        ]

        if crop.size == 0:

            return None

        return cv2.copyMakeBorder(
            crop,
            15,
            15,
            15,
            15,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )


# --------------------------------------------------
# DECODE QR AND RETURN CROPPED IMAGE
# --------------------------------------------------

def decode_qr(image_path):

    img = cv2.imread(
        image_path
    )

    if img is None:

        return None, None

    decoded = zbar_decode(
        img
    )

    if not decoded:

        return None, None

    # First detected QR code

    qr = decoded[0]

    payload = qr.data.decode(
        "utf-8",
        errors="ignore"
    )

    qr_crop = rectify_qr(
        img,
        qr
    )

    return payload, qr_crop


# --------------------------------------------------
# SAVE UPLOADED OR CAMERA IMAGE
# --------------------------------------------------

def save_uploaded_image(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as tmp:

        tmp.write(
            uploaded_file.getvalue()
        )

        return tmp.name


# --------------------------------------------------
# INPUT METHOD
# --------------------------------------------------

source = st.radio(
    "Input method",
    [
        "Upload image",
        "Use webcam"
    ]
)


image_path = None


if source == "Upload image":

    uploaded = st.file_uploader(
        "Upload a QR code image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded is not None:

        image_path = save_uploaded_image(
            uploaded
        )


else:

    cam_image = st.camera_input(
        "Scan a QR code"
    )

    if cam_image is not None:

        image_path = save_uploaded_image(
            cam_image
        )


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

if image_path:

    try:

        # ------------------------------------------
        # DISPLAY ORIGINAL IMAGE
        # ------------------------------------------

        st.image(
            image_path,
            caption="Original input image",
            width=300
        )


        # ------------------------------------------
        # QR DETECTION
        # ------------------------------------------

        payload, qr_crop = decode_qr(
            image_path
        )


        if payload is None:

            st.error(
                "Could not decode a QR code in this image."
            )


        elif qr_crop is None:

            st.error(
                "QR code decoded, but its region could not be cropped."
            )


        else:

            st.success(
                "QR code detected successfully."
            )


            # --------------------------------------
            # DISPLAY DECODED PAYLOAD
            # --------------------------------------

            st.subheader(
                "Decoded QR Payload"
            )

            st.code(
                payload,
                language=None
            )


            # --------------------------------------
            # DISPLAY CROPPED QR
            # --------------------------------------

            st.subheader(
                "Detected QR Region"
            )

            st.image(
                cv2.cvtColor(
                    qr_crop,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Cropped and perspective-corrected QR",
                width=300
            )


            # --------------------------------------
            # BRANCH B: PAYLOAD ANALYSIS
            # --------------------------------------

            payload_result = None

            image_risk_score = None

            heatmap_overlay = None

            cnn_model = None


            with st.spinner(
                "Analyzing QR code..."
            ):

                try:

                    payload_result = analyze_payload(
                        payload
                    )

                except Exception as e:

                    st.error(
                        f"Payload analysis failed: {e}"
                    )

                    payload_result = {
                        "risk_score": 0.0,
                        "reasons": [
                            "Payload analysis could not be completed."
                        ],
                        "payload_type": "Unknown",
                        "blacklisted": False,
                        "blacklist_checked": False
                    }


                # ----------------------------------
                # BRANCH A: CNN IMAGE ANALYSIS
                # ----------------------------------

                cnn_model = load_cnn_model()


                if cnn_model is not None:

                    crop_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".png"
                        ) as tmp:

                            crop_path = tmp.name


                        cv2.imwrite(
                            crop_path,
                            qr_crop
                        )


                        image_risk_score, heatmap_overlay = (
                            predict_with_gradcam(
                                cnn_model,
                                crop_path
                            )
                        )


                    except Exception as e:

                        st.warning(
                            "CNN image analysis failed. "
                            "Continuing with payload analysis."
                        )

                        image_risk_score = None

                        heatmap_overlay = None


                    finally:

                        if (
                            crop_path is not None
                            and os.path.exists(crop_path)
                        ):

                            os.remove(
                                crop_path
                            )


                else:

                    st.warning(
                        "CNN model is unavailable. "
                        "Continuing with payload analysis."
                    )


                # ----------------------------------
                # FUSION
                # ----------------------------------

                result = fuse(
                    image_risk_score,
                    payload_result
                )


            # --------------------------------------
            # FINAL RISK VERDICT
            # --------------------------------------

            st.divider()

            risk_band = result.get(
                "risk_band",
                "Unknown"
            )

            risk_score = result.get(
                "final_score",
                0.0
            )


            band_color = {
                "Low": "green",
                "Medium": "orange",
                "High": "red"
            }.get(
                risk_band,
                "gray"
            )


            st.markdown(
                f"## Risk Level: :{band_color}[{risk_band}]"
            )


            # --------------------------------------
            # RISK PROGRESS BAR
            # --------------------------------------

            st.progress(
                min(
                    max(
                        float(risk_score),
                        0.0
                    ),
                    1.0
                )
            )


            # --------------------------------------
            # SCORE INFORMATION
            # --------------------------------------

            st.write(
                f"**Final Risk Score:** "
                f"{risk_score:.2f}"
            )


            st.write(
                f"**Payload Type:** "
                f"{result.get('payload_type', 'Unknown')}"
            )


            st.write(
                f"**Image Risk Score:** "
                f"{result.get('image_risk_score', 'Unavailable')}"
            )


            st.write(
                f"**Payload Risk Score:** "
                f"{result.get('payload_risk_score', 'Unavailable')}"
            )


            # --------------------------------------
            # IMAGE ANALYSIS STATUS
            # --------------------------------------

            if image_risk_score is None:

                st.info(
                    "CNN image analysis was unavailable. "
                    "The final verdict is based on payload analysis."
                )

            else:

                st.success(
                    "CNN image analysis completed successfully."
                )


            # --------------------------------------
            # EXPLANATION
            # --------------------------------------

            st.subheader(
                "Why this verdict?"
            )


            reasons = result.get(
                "reasons",
                []
            )


            if reasons:

                for reason in reasons:

                    st.write(
                        f"- {reason}"
                    )

            else:

                st.write(
                    "No additional risk reasons were provided."
                )


            # --------------------------------------
            # BLACKLIST STATUS
            # --------------------------------------

            blacklist_status = result.get(
                "blacklist_status"
            )


            if blacklist_status:

                st.subheader(
                    "Blacklist Status"
                )


                if blacklist_status == "listed":

                    st.error(
                        "This payload was found in the blacklist."
                    )

                elif blacklist_status == "not_found":

                    st.success(
                        "This payload was not found in the checked blacklist."
                    )

                elif blacklist_status == "unavailable":

                    st.warning(
                        "Blacklist verification was unavailable."
                    )

                elif blacklist_status == "not_applicable":

                    st.info(
                        "Blacklist check is not applicable to this payload."
                    )


            # --------------------------------------
            # GRAD-CAM HEATMAP
            # --------------------------------------

            if heatmap_overlay is not None:

                st.subheader(
                    "Image Anomaly Heatmap"
                )


                st.image(
                    heatmap_overlay,
                    caption=(
                        "Grad-CAM visualization highlighting "
                        "image regions that influenced the CNN prediction."
                    ),
                    width=300
                )


    finally:

        # ------------------------------------------
        # CLEANUP TEMPORARY ORIGINAL IMAGE
        # ------------------------------------------

        if (
            image_path is not None
            and os.path.exists(image_path)
        ):

            os.remove(
                image_path
            )
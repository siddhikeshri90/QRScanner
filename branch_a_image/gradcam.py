"""
gradcam.py

CNN inference + gradient-based Grad-CAM for the QR tampering classifier.

Works with the saved MobileNetV2 transfer-learning model
(MobileNetV2 nested inside a functional model, sigmoid output).

Class mapping (same as train_cnn.py):
    0 = Genuine, 1 = Tampered. The model output is P(tampered).

Preprocessing is identical to train_cnn.py:
    RGB -> tf.image.resize (bilinear) to 224x224 -> x / 127.5 - 1

Public functions:
    analyze_image(model, image)  -> dict with score, overlay, status
    predict_with_gradcam(model, image) -> (score, overlay or None)

`image` is a BGR numpy array (OpenCV) or an image file path.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

DEFAULT_IMG_SIZE = (224, 224)
SCORE_MATCH_TOLERANCE = 0.02


# --------------------------------------------------
# INPUT HANDLING
# --------------------------------------------------

def _model_input_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]

        height, width = shape[1], shape[2]

        if isinstance(height, int) and isinstance(width, int):
            return (height, width)

    except Exception:
        pass

    return DEFAULT_IMG_SIZE


def _to_rgb_uint8(image):
    if isinstance(image, (str, os.PathLike)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)

        if bgr is None:
            raise ValueError("Could not read the image file.")

    else:
        bgr = np.asarray(image)

    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Image must have 3 colour channels.")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(image, size=DEFAULT_IMG_SIZE):
    """
    Same pipeline as training:
    tf.image.resize (bilinear) then x / 127.5 - 1.

    Returns:
        model_batch
        resized_rgb_uint8
    """

    rgb = _to_rgb_uint8(image)

    resized = tf.image.resize(
        tf.convert_to_tensor(rgb),
        size
    )

    resized = tf.cast(resized, tf.float32)

    batch = tf.expand_dims(
        resized / 127.5 - 1.0,
        axis=0
    )

    display = np.clip(
        resized.numpy(),
        0,
        255
    ).astype(np.uint8)

    return batch, display


def _as_float(tensor):
    value = tensor.numpy() if hasattr(tensor, "numpy") else tensor

    return float(
        np.asarray(value).reshape(-1)[0]
    )


# --------------------------------------------------
# LAYER LOOKUP
# --------------------------------------------------

def _call_layer(layer, x):
    try:
        return layer(x, training=False)

    except TypeError:
        return layer(x)


def _find_base_model(model):
    """
    Finds the nested convolutional base (MobileNetV2).
    """

    for index, layer in enumerate(model.layers):

        if isinstance(layer, tf.keras.Model):

            try:
                if len(layer.output.shape) == 4:
                    return index, layer

            except Exception:
                continue

    return None, None


def _gradients_nested(model, batch, base_index, base):
    """
    Gradient of the tampered logit with respect to the last
    convolutional feature map.

    Using the logit avoids zero gradients when sigmoid saturates.
    """

    head = [
        layer
        for layer in model.layers[base_index + 1:]
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]

    use_logit = (
        bool(head)
        and isinstance(head[-1], tf.keras.layers.Dense)
        and head[-1].units == 1
        and getattr(
            head[-1].activation,
            "__name__",
            ""
        ) == "sigmoid"
    )

    with tf.GradientTape() as tape:

        conv = base(batch, training=False)

        tape.watch(conv)

        x = conv

        for layer in (
            head[:-1] if use_logit else head
        ):
            x = _call_layer(layer, x)

        if use_logit:

            kernel, bias = head[-1].get_weights()

            logit = (
                tf.matmul(x, tf.constant(kernel))
                + tf.constant(bias)
            )

            target = logit

            probability = tf.sigmoid(logit)

        else:

            target = x

            probability = x

        target_value = target[:, 0]

    grads = tape.gradient(
        target_value,
        conv
    )

    return conv, grads, _as_float(probability)


def _gradients_flat(model, batch):
    """
    Fallback for models without a nested base model.
    """

    conv_layer = None

    for layer in reversed(model.layers):

        try:

            if len(layer.output.shape) == 4:

                conv_layer = layer

                break

        except Exception:
            continue

    if conv_layer is None:
        raise ValueError(
            "No convolutional feature layer found."
        )

    grad_model = tf.keras.Model(
        model.inputs,
        [
            conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv, predictions = grad_model(
            batch,
            training=False
        )

        tape.watch(conv)

        target_value = predictions[:, 0]

    grads = tape.gradient(
        target_value,
        conv
    )

    return conv, grads, _as_float(predictions)


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def _heatmap(conv, grads):

    if grads is None:
        raise RuntimeError(
            "Gradients could not be computed."
        )

    weights = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    cam = tf.reduce_sum(
        conv[0] * weights,
        axis=-1
    )

    cam = tf.nn.relu(cam).numpy()

    peak = float(cam.max())

    if not np.isfinite(peak):
        raise RuntimeError(
            "Heatmap contains invalid values."
        )

    if peak <= 1e-12:

        return np.zeros_like(cam), False

    return cam / peak, True


def overlay_heatmap(
    heatmap,
    rgb_image,
    alpha=0.4
):

    height, width = rgb_image.shape[:2]

    heat = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    heat = np.uint8(
        255 * np.clip(
            heat,
            0.0,
            1.0
        )
    )

    colour = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET
    )

    base_bgr = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

    blended = cv2.addWeighted(
        base_bgr,
        1 - alpha,
        colour,
        alpha,
        0
    )

    return cv2.cvtColor(
        blended,
        cv2.COLOR_BGR2RGB
    )


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def analyze_image(model, image):
    """
    Returns a dict:

        score        - P(tampered), from the model's own forward pass
        overlay      - RGB Grad-CAM overlay, or None
        gradcam_ok   - True if the heatmap was produced and verified
        gradcam_note - short status text

    Raises only if the CNN prediction itself fails.
    """

    size = _model_input_size(model)

    batch, display = preprocess(
        image,
        size
    )

    output = model(
        batch,
        training=False
    )

    score = _as_float(output)

    if (
        not np.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError(
            "CNN returned an invalid score."
        )

    result = {
        "score": score,
        "overlay": None,
        "gradcam_ok": False,
        "gradcam_note": "",
    }

    try:

        base_index, base = _find_base_model(model)

        if base is not None:

            conv, grads, manual_probability = (
                _gradients_nested(
                    model,
                    batch,
                    base_index,
                    base
                )
            )

        else:

            conv, grads, manual_probability = (
                _gradients_flat(
                    model,
                    batch
                )
            )

        # Heatmap must come from the same prediction
        # we report.

        if (
            abs(manual_probability - score)
            > SCORE_MATCH_TOLERANCE
        ):

            raise RuntimeError(
                "heatmap pass does not match model output"
            )

        heat, has_signal = _heatmap(
            conv,
            grads
        )

        result["overlay"] = overlay_heatmap(
            heat,
            display
        )

        result["gradcam_ok"] = True

        result["gradcam_note"] = (
            ""
            if has_signal
            else "No region increased the tampered score."
        )

    except Exception as exc:

        result["gradcam_note"] = (
            f"Grad-CAM unavailable "
            f"({type(exc).__name__})."
        )

    return result


def predict_with_gradcam(model, image):
    """
    Backward-compatible wrapper.

    Returns:
        (score, overlay or None)
    """

    info = analyze_image(
        model,
        image
    )

    return (
        info["score"],
        info["overlay"]
    )


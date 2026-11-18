"""
generate_dataset.py

Builds image dataset for Branch A (CNN).

GENUINE:
- Normal black and white QR codes
- Blue, green, purple, maroon QR codes
- Different QR payloads
- Photo-like augmentations

TAMPERED:
- Sticker overlay QR replacement
- Coloured and black-and-white base QR
- Synthetic module damage

Run from project root:
python data/generate_dataset.py --n_genuine 600 --n_tampered 600
"""

import os
import random
import argparse

import qrcode
import cv2
import numpy as np
from PIL import Image

import albumentations as A


# ---------------- RANDOM SEED ----------------

random.seed(42)
np.random.seed(42)


# ---------------- DIRECTORIES ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "images")

GENUINE_DIR = os.path.join(OUT_DIR, "genuine")
TAMPERED_DIR = os.path.join(OUT_DIR, "tampered")

IMG_SIZE = 400


# ---------------- COLOUR PALETTE ----------------

# Dark foreground + light background.
# These combinations maintain good QR contrast.

COLOUR_PAIRS = [
    ((23, 54, 93), (255, 255, 255)),       # Blue
    ((23, 77, 53), (255, 255, 255)),       # Green
    ((84, 36, 94), (255, 255, 255)),       # Purple
    ((122, 32, 32), (255, 255, 255)),      # Maroon
    ((23, 59, 112), (255, 253, 242)),      # Navy
    ((22, 78, 99), (240, 250, 250)),       # Teal
    ((70, 70, 70), (255, 242, 216)),       # Dark grey
]


# ---------------- PAYLOAD GENERATORS ----------------

def random_url(benign=True):

    domains_benign = [
        "amazon.in",
        "irctc.co.in",
        "flipkart.com",
        "hdfcbank.com",
        "gmail.com",
        "instagram.com",
        "bookmyshow.com"
    ]

    domains_scam = [
        "paypa1-verify.com",
        "196.11.23.4",
        "hdfcbannk-secure.tk",
        "bit.ly"
    ]

    domain = random.choice(
        domains_benign if benign else domains_scam
    )

    if not benign and domain == "196.11.23.4":
        return "http://196.11.23.4/login"

    if not benign and domain == "bit.ly":
        return "https://bit.ly/xk93jd"

    path = random.choice([
        "",
        "/login",
        "/verify-account",
        "/track",
        "/pay?amt=500"
    ])

    return f"https://{domain}{path}"


def random_upi(benign=True):

    if benign:

        pa = random.choice([
            "merchant.store@okhdfcbank",
            "ravi.traders@okaxis",
            "citycafe@oksbi"
        ])

        amount = ""

    else:

        pa = random.choice([
            "refund.claim@oksbi",
            "parkingfine123@okicici",
            "9876543210@paytm"
        ])

        amount = "&am=499"

    return f"upi://pay?pa={pa}&pn=Merchant{amount}&cu=INR"


def random_text(benign=True):

    if benign:

        return random.choice([
            "Table 12 - Cafe Menu",
            "Asset ID: LAP-2291",
            "Wifi guest network information",
            "Movie ticket information",
            "Event entry pass",
            "Product information"
        ])

    return random.choice([
        "URGENT: Your account is suspended, verify now",
        "Claim your prize now, limited time",
        "KYC update required immediately or account blocked",
        "Your payment failed, click to claim refund"
    ])


def random_payload(benign=True):

    kind = random.choice([
        "url",
        "upi",
        "text"
    ])

    if kind == "url":
        return random_url(benign)

    elif kind == "upi":
        return random_upi(benign)

    return random_text(benign)


# ---------------- QR IMAGE GENERATION ----------------

def make_qr_image(payload, size_px=IMG_SIZE, colored=False):

    """
    Creates a valid QR image.

    colored=False:
        Normal black and white QR.

    colored=True:
        Coloured QR using a high-contrast colour palette.

    Colour is applied directly to the QR pixels.
    No black-and-white fallback is used.
    """

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )

    qr.add_data(payload)
    qr.make(fit=True)

    # First generate a standard QR matrix.
    img = qr.make_image(
        fill_color="black",
        back_color="white"
    ).convert("RGB")

    img = img.resize(
        (size_px, size_px),
        Image.Resampling.NEAREST
    )

    # Apply actual RGB colours to the QR pixels.
    if colored:

        foreground, background = random.choice(COLOUR_PAIRS)

        pixels = np.array(img)

        # Identify dark and light QR pixels.
        dark_mask = np.mean(pixels, axis=2) < 128

        coloured_pixels = np.zeros_like(pixels)

        coloured_pixels[:] = background
        coloured_pixels[dark_mask] = foreground

        return coloured_pixels.astype(np.uint8)

    # Standard black and white QR.
    return np.array(img)


# ---------------- STICKER OVERLAY ----------------

def paste_sticker_overlay(base, overlay_payload):

    """
    Simulates a suspicious QR sticker placed over
    an original QR code.
    """

    h, w, _ = base.shape

    overlay_size = int(
        w * random.uniform(0.5, 0.8)
    )

    # Fake sticker is generated in black and white.
    overlay = make_qr_image(
        overlay_payload,
        size_px=overlay_size,
        colored=False
    )

    oh, ow, _ = overlay.shape

    angle = random.uniform(-8, 8)

    matrix = cv2.getRotationMatrix2D(
        (ow // 2, oh // 2),
        angle,
        1.0
    )

    overlay = cv2.warpAffine(
        overlay,
        matrix,
        (ow, oh),
        borderValue=(255, 255, 255)
    )

    x = random.randint(0, w - ow)
    y = random.randint(0, h - oh)

    result = base.copy()

    result[y:y + oh, x:x + ow] = overlay

    # Sticker boundary shadow.
    shadow = np.zeros_like(result)

    cv2.rectangle(
        shadow,
        (x, y),
        (x + ow, y + oh),
        (40, 40, 40),
        thickness=6
    )

    result = cv2.addWeighted(
        result,
        1.0,
        shadow,
        0.15,
        0
    )

    return result


# ---------------- PHOTO AUGMENTATION ----------------

PHOTO_REALISM = A.Compose([

    A.RandomBrightnessContrast(
        p=0.4
    ),

    A.GaussianBlur(
        blur_limit=(1, 3),
        p=0.3
    ),

    A.ImageCompression(
        quality_range=(60, 95),
        p=0.3
    ),

    A.Rotate(
        limit=5,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(255, 255, 255),
        p=0.3
    ),

    A.RandomShadow(
        p=0.2
    )

])


# ---------------- CLEAN OLD GENERATED IMAGES ----------------

def clean_old_images():

    """
    Removes old generated PNG files so that
    previous datasets do not mix with new ones.
    """

    for folder in [GENUINE_DIR, TAMPERED_DIR]:

        os.makedirs(folder, exist_ok=True)

        for filename in os.listdir(folder):

            if filename.lower().endswith(".png"):

                file_path = os.path.join(
                    folder,
                    filename
                )

                os.remove(file_path)


# ---------------- DATASET GENERATION ----------------

def generate(n_genuine, n_tampered):

    os.makedirs(GENUINE_DIR, exist_ok=True)
    os.makedirs(TAMPERED_DIR, exist_ok=True)

    # Clear old generated images.
    clean_old_images()

    genuine_coloured = 0
    genuine_normal = 0

    tampered_coloured_base = 0
    tampered_normal_base = 0

    print("\nGenerating genuine QR images...")

    for i in range(n_genuine):

        payload = random_payload(benign=True)

        # Guaranteed 50% normal and 50% coloured.
        colored = (i % 2 == 1)

        img = make_qr_image(
            payload,
            colored=colored
        )

        if colored:
            genuine_coloured += 1
        else:
            genuine_normal += 1

        img = PHOTO_REALISM(
            image=img
        )["image"]

        output_path = os.path.join(
            GENUINE_DIR,
            f"genuine_{i:04d}.png"
        )

        Image.fromarray(img).save(output_path)

        if (i + 1) % 100 == 0:

            print(
                f"Genuine: {i + 1}/{n_genuine}"
            )

    print("\nGenerating tampered QR images...")

    for i in range(n_tampered):

        base_payload = random_payload(benign=True)

        malicious_payload = random_payload(benign=False)

        # Both normal and coloured base QR codes.
        colored = (i % 2 == 1)

        base = make_qr_image(
            base_payload,
            colored=colored
        )

        if colored:
            tampered_coloured_base += 1
        else:
            tampered_normal_base += 1

        if random.random() < 0.7:

            # Sticker replacement.
            img = paste_sticker_overlay(
                base,
                malicious_payload
            )

        else:

            # Synthetic localised module damage.
            img = base.copy()

            h, w, _ = img.shape

            patch_size = random.randint(20, 55)

            x = random.randint(
                30,
                w - patch_size - 30
            )

            y = random.randint(
                30,
                h - patch_size - 30
            )

            cv2.rectangle(
                img,
                (x, y),
                (x + patch_size, y + patch_size),
                (random.randint(0, 80),) * 3,
                thickness=-1
            )

        img = PHOTO_REALISM(
            image=img
        )["image"]

        output_path = os.path.join(
            TAMPERED_DIR,
            f"tampered_{i:04d}.png"
        )

        Image.fromarray(img).save(output_path)

        if (i + 1) % 100 == 0:

            print(
                f"Tampered: {i + 1}/{n_tampered}"
            )

    # ---------------- FINAL REPORT ----------------

    print("\n================================")
    print("       DATASET GENERATED")
    print("================================")

    print(f"\nGenuine total: {n_genuine}")
    print(f"  Normal black-white: {genuine_normal}")
    print(f"  Coloured: {genuine_coloured}")

    print(f"\nTampered total: {n_tampered}")
    print(f"  Normal base: {tampered_normal_base}")
    print(f"  Coloured base: {tampered_coloured_base}")

    print("\nGenuine folder:")
    print(GENUINE_DIR)

    print("\nTampered folder:")
    print(TAMPERED_DIR)

    print("\nDataset generation completed successfully!")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_genuine",
        type=int,
        default=600
    )

    parser.add_argument(
        "--n_tampered",
        type=int,
        default=600
    )

    args = parser.parse_args()

    if args.n_genuine < 1 or args.n_tampered < 1:

        parser.error(
            "Image counts must be positive."
        )

    generate(
        args.n_genuine,
        args.n_tampered
    )
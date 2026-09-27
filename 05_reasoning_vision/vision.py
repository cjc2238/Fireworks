"""Describe an image from a URL, or from a local file sent as base64.

Usage:  python 05_reasoning_vision/vision.py                 # uses a sample URL
        python 05_reasoning_vision/vision.py photo.jpg       # uses your file
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import base64
import mimetypes

from fwlearn import VISION_MODEL, client

SAMPLE_URL = "https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?w=800"

if len(sys.argv) > 1:
    path = pathlib.Path(sys.argv[1])
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    print(f"Sending {path.name} as base64 ({path.stat().st_size / 1e6:.2f} MB)")
else:
    url = SAMPLE_URL
    print("Sending sample URL:", url)

resp = client().chat.completions.create(
    model=VISION_MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in 3 bullet points, then list any text you can read."},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }],
    max_tokens=500,
)
print(resp.choices[0].message.content)
print(f"\nprompt_tokens={resp.usage.prompt_tokens}  (images cost tokens too)")

import base64
import io
import json
import os
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "image_details.db")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
DATABRICKS_BASE_URL = os.getenv("DATABRICKS_BASE_URL", "").rstrip("/")
DATABRICKS_MODEL = os.getenv("DATABRICKS_MODEL", "databricks-claude-sonnet-4-6")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "").strip()
HUGGINGFACE_TEXT_TO_IMAGE_URL = os.getenv(
    "HUGGINGFACE_TEXT_TO_IMAGE_URL",
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
).strip()
HUGGINGFACE_TEXT_TO_IMAGE_MODEL_LABEL = os.getenv(
    "HUGGINGFACE_TEXT_TO_IMAGE_MODEL_LABEL", "stabilityai/stable-diffusion-2-1"
).strip()
HUGGINGFACE_ROUTER_BASE_URL = os.getenv(
    "HUGGINGFACE_ROUTER_BASE_URL", "https://router.huggingface.co/hf-inference/models"
).rstrip("/")
DATABRICKS_OCR_PROMPT = os.getenv("DATABRICKS_OCR_PROMPT","You are an OCR engine. Extract all readable text from the whole image exactly as written. Do not omit text. Return the final OCR result as a single line with spaces instead of line breaks. Do not summarize or explain. If no text is visible, return exactly: NO_TEXT_DETECTED.",)
DATABRICKS_IMAGE_DESCRIPTION_PROMPT = os.getenv("DATABRICKS_IMAGE_DESCRIPTION_PROMPT","Describe the image in 100 words")
DATABRICKS_PROMPT_GENERATION_PROMPT = os.getenv("DATABRICKS_PROMPT_GENERATION_PROMPT","Using the OCR text and the visual description, write a descriptive prompt that clearly explains the image content, visible text, and likely intent. Keep it concise but informative. Return plain text only.",)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Tools</title>
    <style>
        :root {
            --bg: #f7f2e7;
            --panel: rgba(255, 252, 246, 0.94);
            --ink: #1f2a2c;
            --muted: #5d676a;
            --accent: #1e6f5c;
            --accent-dark: #164f42;
            --line: rgba(31, 42, 44, 0.12);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(30, 111, 92, 0.18), transparent 28%),
                radial-gradient(circle at right center, rgba(167, 201, 87, 0.18), transparent 24%),
                linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
        }

        .shell {
            width: min(980px, calc(100% - 32px));
            margin: 32px auto;
            padding: 28px;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--panel);
            box-shadow: 0 18px 60px rgba(20, 34, 28, 0.08);
        }

        h1 {
            margin: 0 0 8px;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1;
        }

        p {
            margin: 0 0 20px;
            color: var(--muted);
        }

        form {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }

        .tab-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 16px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.8);
            color: var(--ink);
            text-decoration: none;
            font-weight: 700;
        }

        .tab-link.active {
            background: var(--accent);
            color: #fff;
            border-color: transparent;
        }

        .panel {
            padding: 18px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.66);
            margin-bottom: 18px;
        }

        .panel.hidden {
            display: none;
        }

        input[type="file"] {
            min-width: 280px;
            padding: 10px;
            border: 1px dashed var(--line);
            border-radius: 12px;
            background: #fff;
        }

        input[type="text"] {
            min-width: 280px;
            padding: 12px 14px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #fff;
            color: var(--ink);
            font: inherit;
        }

        button {
            padding: 12px 18px;
            border: 0;
            border-radius: 999px;
            background: var(--accent);
            color: #fff;
            font-weight: 700;
            cursor: pointer;
        }

        button:hover {
            background: var(--accent-dark);
        }

        .error {
            margin-bottom: 16px;
            padding: 14px 16px;
            border-radius: 14px;
            background: rgba(180, 48, 48, 0.08);
            color: #8a1d1d;
            border: 1px solid rgba(180, 48, 48, 0.16);
        }

        .meta {
            margin-bottom: 16px;
            color: var(--muted);
        }

        .status {
            margin-bottom: 16px;
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid rgba(30, 111, 92, 0.16);
            background: rgba(30, 111, 92, 0.08);
            color: var(--accent-dark);
        }

        .label {
            margin: 16px 0 8px;
            font-weight: 700;
            color: var(--ink);
        }

        .prompt-box {
            min-height: 140px;
        }

        .generated-image {
            display: block;
            width: 100%;
            max-width: 640px;
            margin-top: 12px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: #fff;
            box-shadow: 0 18px 40px rgba(20, 34, 28, 0.08);
        }

        textarea {
            width: 100%;
            min-height: 240px;
            padding: 14px;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: #fff;
            color: var(--ink);
            resize: vertical;
            font: inherit;
        }

        @media (max-width: 700px) {
            .shell {
                width: calc(100% - 20px);
                margin: 10px auto;
                padding: 18px;
            }

            input[type="file"],
            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <h3>Image Tools</h3>
        <p>Upload an image to extract image details and save them to the database, search by image name, or generate an image from the prompt you enter in the text box.</p>

        <div class="tabs">
            <a class="tab-link {% if active_tab == 'upload' %}active{% endif %}" href="/?tab=upload">Upload And Save</a>
            <a class="tab-link {% if active_tab == 'lookup' %}active{% endif %}" href="/?tab=lookup">Lookup From DB</a>
            <a class="tab-link {% if active_tab == 'generate' %}active{% endif %}" href="/?tab=generate">Text To Image</a>
        </div>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if status %}
        <div class="status">{{ status }}</div>
        {% endif %}

        <section class="panel {% if active_tab != 'upload' %}hidden{% endif %}">
            <form method="post" enctype="multipart/form-data">
                <input type="hidden" name="form_action" value="upload">
                <input type="text" name="image_name" placeholder="Enter image name to save" value="{{ image_name }}" required>
                <input type="file" name="image_file" accept="image/*" required>
                <button type="submit">Extract And Save</button>
            </form>
        </section>

        <section class="panel {% if active_tab != 'lookup' %}hidden{% endif %}">
            <form method="post">
                <input type="hidden" name="form_action" value="lookup">
                <input type="text" name="lookup_image_name" placeholder="Enter image name to search in DB" value="{{ lookup_image_name }}" required>
                <button type="submit">Get From DB</button>
            </form>
        </section>

        <section class="panel {% if active_tab != 'generate' %}hidden{% endif %}">
            <form method="post">
                <input type="hidden" name="form_action" value="generate">
                <textarea class="prompt-box" name="text_to_image_prompt" placeholder="Enter the prompt used to generate the image" required>{{ text_to_image_prompt }}</textarea>
                <button type="submit">Generate Image</button>
            </form>

            {% if generated_image_data_url %}
            <div class="meta">
                <strong>Model:</strong> {{ text_to_image_model }}
            </div>
            <div class="label">Prompt</div>
            <textarea readonly>{{ text_to_image_prompt }}</textarea>
            <div class="label">Generated Image</div>
            <img class="generated-image" src="{{ generated_image_data_url }}" alt="Generated image from prompt">
            {% endif %}
        </section>

        {% if image_details %}
        <div class="meta">
            <strong>Image Name:</strong> {{ image_name or file_name }}
        </div>
        <p><strong>Image Details</strong></p>
        <textarea readonly>{{ image_details }}</textarea>
        {% endif %}
    </main>
</body>
</html>
"""


def allowed_file(filename):
    extension = os.path.splitext(filename or "")[1].lower()
    return extension in ALLOWED_EXTENSIONS


def init_sqlite_db():
    connection = sqlite3.connect(SQLITE_DB_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_details (
                image_name TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                image_details TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def save_image_details_to_db(image_name, file_name, image_details):
    connection = sqlite3.connect(SQLITE_DB_PATH)
    try:
        connection.execute(
            """
            INSERT INTO image_details (image_name, file_name, image_details)
            VALUES (?, ?, ?)
            """,
            (image_name, file_name, image_details),
        )
        connection.commit()
    finally:
        connection.close()


def get_image_details_from_db(image_name):
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT image_name, file_name, image_details FROM image_details WHERE image_name = ?",
            (image_name,),
        ).fetchone()
    finally:
        connection.close()

    return dict(row) if row else None


init_sqlite_db()


def get_databricks_client():
    if not DATABRICKS_TOKEN:
        raise RuntimeError(
            "DATABRICKS_TOKEN is missing. Add your Databricks PATH to the .env file before running the app."
        )

    return OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=DATABRICKS_BASE_URL,
    )


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.filename or "image.bin")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
    uploaded_file.save(temp_path)
    return temp_path


def extract_text_from_databricks_response(response):
    choices = getattr(response, "choices", []) or []
    if not choices:
        raise RuntimeError("Databricks returned no completion choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("Databricks returned a choice without a message.")

    content = getattr(message, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_segments = []
        for item in content:
            text_value = getattr(item, "text", None)
            if text_value:
                text_segments.append(text_value)
                continue

            if isinstance(item, dict) and item.get("text"):
                text_segments.append(item["text"])

        combined_text = "\n".join(text_segments).strip()
        if combined_text:
            return combined_text

    raise RuntimeError("Databricks returned no text content.")


def image_file_to_base64_png(file_path):
    with Image.open(file_path) as image:
        normalized_image = image.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as normalized_file:
            normalized_path = normalized_file.name

        try:
            normalized_image.save(normalized_path, format="PNG")
            with open(normalized_path, "rb") as image_file:
                image_bytes = image_file.read()
        finally:
            if os.path.exists(normalized_path):
                os.remove(normalized_path)

    return base64.b64encode(image_bytes).decode("utf-8")


def normalize_single_line(text):
    return re.sub(r"\s+", " ", text).strip()


def image_bytes_to_data_url(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        normalized_image = image.convert("RGB")
        with io.BytesIO() as output_buffer:
            normalized_image.save(output_buffer, format="PNG")
            encoded_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded_image}"


def get_huggingface_router_url():
    return f"{HUGGINGFACE_ROUTER_BASE_URL}/{HUGGINGFACE_TEXT_TO_IMAGE_MODEL_LABEL}"


def perform_huggingface_image_request(request_url, request_body):
    api_request = urllib.request.Request(
        request_url,
        data=request_body,
        headers={
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        },
        method="POST",
    )

    with urllib.request.urlopen(api_request, timeout=300) as response:
        return response.read(), response.headers.get("Content-Type", "")


def generate_image_from_huggingface(prompt_text):
    if not HUGGINGFACE_TOKEN:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is missing. Add your Hugging Face token to the .env file before generating images."
        )

    request_body = json.dumps(
        {
            "inputs": prompt_text+ """Carefully review every point
                Remove ambiguity and redundancy
                Rewrite it in a concise, clear, AI‑friendly format
                Optimize wording so image models understand style, subject, lighting, composition, and details clearly

        """, 
            "options": {"wait_for_model": True},
        }
    ).encode("utf-8")

    try:
        response_bytes, response_content_type = perform_huggingface_image_request(
            HUGGINGFACE_TEXT_TO_IMAGE_URL,
            request_body,
        )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            try:
                response_bytes, response_content_type = perform_huggingface_image_request(
                    get_huggingface_router_url(),
                    request_body,
                )
            except urllib.error.HTTPError as retry_exc:
                retry_error_body = retry_exc.read().decode("utf-8", errors="replace")
                try:
                    retry_error_payload = json.loads(retry_error_body)
                    retry_error_message = retry_error_payload.get("error") or retry_error_body
                except json.JSONDecodeError:
                    retry_error_message = retry_error_body or str(retry_exc)
                raise RuntimeError(f"Hugging Face request failed: {retry_error_message}") from retry_exc
            except urllib.error.URLError as retry_exc:
                raise RuntimeError(f"Unable to reach Hugging Face: {retry_exc.reason}") from retry_exc
        else:
            try:
                error_payload = json.loads(error_body)
                error_message = error_payload.get("error") or error_body
            except json.JSONDecodeError:
                error_message = error_body or str(exc)
            raise RuntimeError(f"Hugging Face request failed: {error_message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach Hugging Face: {exc.reason}") from exc

    if "application/json" in response_content_type:
        try:
            response_payload = json.loads(response_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Hugging Face returned JSON instead of an image.") from exc
        raise RuntimeError(response_payload.get("error") or "Hugging Face returned no image.")

    return image_bytes_to_data_url(response_bytes)


def call_databricks(message_content, max_output_tokens):
    try:
        response = get_databricks_client().chat.completions.create(
            model=DATABRICKS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            max_tokens=max_output_tokens,
            temperature=0.2,
        )
    except Exception as exc:
        raise RuntimeError(f"Databricks request failed: {exc}") from exc

    return extract_text_from_databricks_response(response)


def extract_ocr_text_from_image(file_path):
    image_base64 = image_file_to_base64_png(file_path)
    extracted_text = call_databricks(
        [
            {"type": "text", "text": DATABRICKS_OCR_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                }
            },
        ],
        max_output_tokens=2500,
    )
    return normalize_single_line(extracted_text) or "NO_TEXT_DETECTED"


def extract_image_details_from_image(file_path):
    image_base64 = image_file_to_base64_png(file_path)
    image_details = call_databricks(
        [
            {"type": "text", "text": DATABRICKS_IMAGE_DESCRIPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                }
            },
        ],
        max_output_tokens=1000,
    )
    return normalize_single_line(image_details)


def generate_prompt_from_ocr_text(extracted_text, image_details):
    prompt_text = call_databricks(
        (
            f"{DATABRICKS_PROMPT_GENERATION_PROMPT}\n\n"
            "Extracted OCR text:\n"
            f"{extracted_text}\n\n"
            "Image details:\n"
            f"{image_details}"
        ),
        max_output_tokens=600,
    )
    return normalize_single_line(prompt_text)


@app.route("/", methods=["GET", "POST"])
def index():
    page_context = {
        "error": None,
        "status": None,
        "active_tab": request.args.get("tab", "upload"),
        "file_name": "",
        "text_to_image_model": HUGGINGFACE_TEXT_TO_IMAGE_MODEL_LABEL,
        "image_name": "",
        "lookup_image_name": "",
        "text_to_image_prompt": "",
        "extracted_text": "",
        "generated_image_data_url": "",
        "image_details": "",
        "generated_prompt": "",
    }

    if request.method == "POST":
        form_action = request.form.get("form_action", "upload")
        if form_action == "lookup":
            page_context["active_tab"] = "lookup"
        elif form_action == "generate":
            page_context["active_tab"] = "generate"
        else:
            page_context["active_tab"] = "upload"

        if form_action == "lookup":
            lookup_image_name = (request.form.get("lookup_image_name") or "").strip()
            page_context["lookup_image_name"] = lookup_image_name

            if not lookup_image_name:
                page_context["error"] = "Enter an image name to search in the database."
            else:
                db_record = get_image_details_from_db(lookup_image_name)
                if db_record:
                    page_context.update(
                        {
                            "image_name": db_record["image_name"],
                            "file_name": db_record["file_name"],
                            "image_details": db_record["image_details"],
                            "status": "Image details loaded from database.",
                        }
                    )
                else:
                    page_context["error"] = "No data available in db."
        elif form_action == "generate":
            text_to_image_prompt = normalize_single_line(
                request.form.get("text_to_image_prompt") or ""
            )
            page_context["text_to_image_prompt"] = text_to_image_prompt

            if not text_to_image_prompt:
                page_context["error"] = "Enter a prompt before generating the image."
            else:
                try:
                    page_context["generated_image_data_url"] = generate_image_from_huggingface(
                        text_to_image_prompt
                    )
                    page_context["status"] = "Image generated from the prompt."
                except Exception as exc:
                    page_context["error"] = f"Image generation failed: {exc}"
        else:
            uploaded_file = request.files.get("image_file")
            image_name = (request.form.get("image_name") or "").strip()
            page_context["image_name"] = image_name

            if not image_name:
                page_context["error"] = "Enter an image name before uploading the file."
            elif get_image_details_from_db(image_name):
                page_context["error"] = (
                    "This image name is already available in db. Choose another name."
                )
            elif uploaded_file is None or uploaded_file.filename == "":
                page_context["error"] = "Select an image file to process."
            elif not allowed_file(uploaded_file.filename):
                page_context["error"] = "Unsupported file type. Upload a common image format."
            else:
                temp_path = save_uploaded_file(uploaded_file)
                try:
                    image_details = extract_image_details_from_image(temp_path)
                    save_image_details_to_db(image_name, uploaded_file.filename, image_details)
                    page_context.update(
                        {
                            "file_name": uploaded_file.filename,
                            "image_name": image_name,
                            "image_details": image_details,
                            "status": "Image details extracted and saved to database.",
                        }
                    )
                except Exception as exc:
                    page_context["error"] = f"Processing failed: {exc}"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    return render_template_string(HTML_TEMPLATE, **page_context)


if __name__ == "__main__":
    app.run(debug=True,port=5004)
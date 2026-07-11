{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ajoshi231/ai-cooking-assistant/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "bbabd8d0",
      "metadata": {
        "id": "bbabd8d0"
      },
      "source": [
        "# 🍳 AI Cooking Assistant\n",
        "### MLH Global Hack Week — Season Kickoff\n",
        "\n",
        "A personalised meal suggester that learns your preferences over time.\n",
        "\n",
        "**What we're building:**\n",
        "1. An onboarding flow that builds your cooking profile\n",
        "2. A meal suggester that reads your profile and asks Gemini for ideas\n",
        "3. A feedback loop that rates suggestions and improves over time\n",
        "\n",
        "**Stack:** Python · Pandas · Gemini API · Gradio\n",
        "\n",
        "---\n",
        "\n",
        "> **Getting started on Google Colab:**\n",
        "> 1. Click the 🔑 key icon in the left sidebar\n",
        "> 2. Add a secret named `GEMINI_API_KEY`\n",
        "> 3. Paste your key from [aistudio.google.com](https://aistudio.google.com) (free, no credit card)\n",
        "> 4. Enable notebook access for the secret\n",
        "> 5. Run cells from top to bottom following along with the stream\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "03d032ee",
      "metadata": {
        "id": "03d032ee"
      },
      "source": [
        "## 0. Setup\n",
        "\n",
        "Install dependencies and configure your free Gemini API key.\n",
        "\n",
        "Get yours free at [aistudio.google.com](https://aistudio.google.com) — sign in with Google, click **Get API key**. No credit card needed.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "fd0ae720",
      "metadata": {
        "id": "fd0ae720",
        "outputId": "a4400e52-0e21-42f0-9ca0-b67ade6d2dfe",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: gradio in /usr/local/lib/python3.12/dist-packages (6.19.0)\n",
            "Requirement already satisfied: google-generativeai in /usr/local/lib/python3.12/dist-packages (0.8.6)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)\n",
            "Requirement already satisfied: google-genai in /usr/local/lib/python3.12/dist-packages (2.10.0)\n",
            "Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (4.14.0)\n",
            "Requirement already satisfied: brotli>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (1.2.0)\n",
            "Requirement already satisfied: fastapi<1.0,>=0.115.2 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.138.0)\n",
            "Requirement already satisfied: gradio-client==2.5.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (2.5.0)\n",
            "Requirement already satisfied: groovy~=0.1 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.1.2)\n",
            "Requirement already satisfied: hf-gradio<1.0,>=0.4.1 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.4.1)\n",
            "Requirement already satisfied: httpx<1.0,>=0.24.1 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.28.1)\n",
            "Requirement already satisfied: huggingface-hub<2.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (1.20.1)\n",
            "Requirement already satisfied: jinja2<4.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (3.1.6)\n",
            "Requirement already satisfied: markupsafe<4.0,>=2.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (3.0.3)\n",
            "Requirement already satisfied: numpy<3.0,>=1.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (2.0.2)\n",
            "Requirement already satisfied: orjson~=3.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (3.11.9)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from gradio) (26.2)\n",
            "Requirement already satisfied: pillow<13.0,>=8.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (11.3.0)\n",
            "Requirement already satisfied: pydantic<=3.0,>=2.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (2.13.4)\n",
            "Requirement already satisfied: pydub<1.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.25.1)\n",
            "Requirement already satisfied: python-multipart<1.0,>=0.0.18 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.0.32)\n",
            "Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.12/dist-packages (from gradio) (2025.2)\n",
            "Requirement already satisfied: pyyaml<7.0,>=5.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (6.0.3)\n",
            "Requirement already satisfied: safehttpx<0.2.0,>=0.1.7 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.1.7)\n",
            "Requirement already satisfied: semantic-version~=2.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (2.10.0)\n",
            "Requirement already satisfied: starlette<2.0,>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from gradio) (1.3.1)\n",
            "Requirement already satisfied: tomlkit<0.15.0,>=0.12.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.14.0)\n",
            "Requirement already satisfied: typer<1.0,>=0.12 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.25.1)\n",
            "Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (4.15.0)\n",
            "Requirement already satisfied: uvicorn>=0.14.0 in /usr/local/lib/python3.12/dist-packages (from gradio) (0.49.0)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from gradio-client==2.5.0->gradio) (2025.3.0)\n",
            "Requirement already satisfied: google-ai-generativelanguage==0.6.15 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (0.6.15)\n",
            "Requirement already satisfied: google-api-core in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.30.3)\n",
            "Requirement already satisfied: google-api-python-client in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.197.0)\n",
            "Requirement already satisfied: google-auth>=2.15.0 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.49.0)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (5.29.6)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (4.67.3)\n",
            "Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /usr/local/lib/python3.12/dist-packages (from google-ai-generativelanguage==0.6.15->google-generativeai) (1.28.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2026.2)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.28.1 in /usr/local/lib/python3.12/dist-packages (from google-genai) (2.32.4)\n",
            "Requirement already satisfied: tenacity<9.2.0,>=8.2.3 in /usr/local/lib/python3.12/dist-packages (from google-genai) (9.1.4)\n",
            "Requirement already satisfied: websockets<17.0,>=13.0.0 in /usr/local/lib/python3.12/dist-packages (from google-genai) (15.0.1)\n",
            "Requirement already satisfied: distro<2,>=1.7.0 in /usr/local/lib/python3.12/dist-packages (from google-genai) (1.9.0)\n",
            "Requirement already satisfied: sniffio in /usr/local/lib/python3.12/dist-packages (from google-genai) (1.3.1)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.12/dist-packages (from anyio<5.0,>=3.0->gradio) (3.18)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from fastapi<1.0,>=0.115.2->gradio) (0.4.2)\n",
            "Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from fastapi<1.0,>=0.115.2->gradio) (0.0.4)\n",
            "Requirement already satisfied: googleapis-common-protos<2.0.0,>=1.63.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core->google-generativeai) (1.75.0)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (0.4.2)\n",
            "Requirement already satisfied: cryptography>=38.0.3 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (49.0.0)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (4.9.1)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1.0,>=0.24.1->gradio) (2026.6.17)\n",
            "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1.0,>=0.24.1->gradio) (1.0.9)\n",
            "Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1.0,>=0.24.1->gradio) (0.16.0)\n",
            "Requirement already satisfied: click>=8.4.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=1.2.0->gradio) (8.4.2)\n",
            "Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=1.2.0->gradio) (3.29.4)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=1.2.0->gradio) (1.5.1)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<=3.0,>=2.0->gradio) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.46.4 in /usr/local/lib/python3.12/dist-packages (from pydantic<=3.0,>=2.0->gradio) (2.46.4)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.28.1->google-genai) (3.4.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.28.1->google-genai) (2.5.0)\n",
            "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer<1.0,>=0.12->gradio) (1.5.4)\n",
            "Requirement already satisfied: rich>=13.8.0 in /usr/local/lib/python3.12/dist-packages (from typer<1.0,>=0.12->gradio) (13.9.4)\n",
            "Requirement already satisfied: httplib2<1.0.0,>=0.19.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.31.2)\n",
            "Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.4.0)\n",
            "Requirement already satisfied: uritemplate<5,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (4.2.0)\n",
            "Requirement already satisfied: cffi>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from cryptography>=38.0.3->google-auth>=2.15.0->google-generativeai) (2.0.0)\n",
            "Requirement already satisfied: grpcio<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.81.1)\n",
            "Requirement already satisfied: grpcio-status<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.71.2)\n",
            "Requirement already satisfied: pyparsing<4,>=3.1 in /usr/local/lib/python3.12/dist-packages (from httplib2<1.0.0,>=0.19.0->google-api-python-client->google-generativeai) (3.3.2)\n",
            "Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in /usr/local/lib/python3.12/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai) (0.6.3)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer<1.0,>=0.12->gradio) (4.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer<1.0,>=0.12->gradio) (2.20.0)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from cffi>=2.0.0->cryptography>=38.0.3->google-auth>=2.15.0->google-generativeai) (3.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer<1.0,>=0.12->gradio) (0.1.2)\n"
          ]
        }
      ],
      "source": [
        "!pip install gradio google-generativeai pandas google-genai\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "id": "cb544ea8",
      "metadata": {
        "id": "cb544ea8",
        "outputId": "5060f193-2703-473d-d63f-53474a155c79",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Running locally — paste your API key above\n",
            "⚠️  No API key found. Add it to Colab secrets or paste it above.\n"
          ]
        }
      ],
      "source": [
        "import gradio as gr\n",
        "import google.generativeai as genai # Corrected import\n",
        "import pandas as pd\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# --- API Key Setup ---\n",
        "# On Google Colab: uses the secrets manager (recommended)\n",
        "# Locally: paste your key directly into the string below\n",
        "\n",
        "try:\n",
        "    from google.colab import userdata\n",
        "    API_KEY = userdata.get('GEMINI_API_KEY')\n",
        "    print(\"✅ Loaded API key from Colab secrets\")\n",
        "except Exception:\n",
        "    API_KEY = \"\"  # TODO: paste your key here if running locally\n",
        "    print(\"Running locally — paste your API key above\")\n",
        "\n",
        "if not API_KEY:\n",
        "    print(\"⚠️  No API key found. Add it to Colab secrets or paste it above.\")\n",
        "else:\n",
        "    genai.configure(api_key=API_KEY) # Configure API key\n",
        "    model = genai.GenerativeModel(\"gemini-1.5-flash\")\n",
        "    print(\"✅ Gemini ready\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "f8a30052",
      "metadata": {
        "id": "f8a30052"
      },
      "source": [
        "## 1. The Profile CSV\n",
        "\n",
        "Before we build the UI, let's think about what we're storing.\n",
        "\n",
        "Every user has a profile that tracks:\n",
        "- Their kitchen equipment\n",
        "- Dietary restrictions and allergies\n",
        "- Time they typically have available\n",
        "- Flavour preferences\n",
        "- Past meal suggestions and how they rated them\n",
        "\n",
        "This is what makes suggestions get better over time — the prompt we send Gemini gets richer with every session.\n",
        "\n",
        "> **Note on Colab:** the CSV lives in your Colab session storage and resets when your session ends. That's fine for this workshop — in a real app you'd save to Google Drive or a database.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "id": "55afbd08",
      "metadata": {
        "id": "55afbd08",
        "outputId": "aa689cb9-836f-4ea0-f094-2fc13f66c7b6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Profile functions ready ✅\n",
            "Existing profile: {'equipment': 'Hob,Oven', 'dietary': nan, 'allergies': 'none', 'time_available': 30, 'spice_level': 'Medium', 'cuisine_prefs': 'Italian,Asian'}\n"
          ]
        }
      ],
      "source": [
        "# Profile file locations\n",
        "PROFILE_PATH = Path(\"data/profile.csv\")\n",
        "HISTORY_PATH = Path(\"data/history.csv\")\n",
        "\n",
        "# Create data directory if it doesn't exist\n",
        "PROFILE_PATH.parent.mkdir(exist_ok=True)\n",
        "\n",
        "def load_profile():\n",
        "    if PROFILE_PATH.exists():\n",
        "      df = pd.read_csv(PROFILE_PATH)\n",
        "      return df.iloc[0].to_dict() if len(df) > 0 else {}\n",
        "    return {}\n",
        "\n",
        "def save_profile(profile: dict):\n",
        "    pd.DataFrame([profile]).to_csv(PROFILE_PATH, index=False)\n",
        "    print(\"Profile saved, your meal is not eternalised as data :)\")\n",
        "\n",
        "def load_history():\n",
        "    if HISTORY_PATH.exists():\n",
        "      return pd.read_csv(HISTORY_PATH)\n",
        "    return pd.DataFrame(columns=[\"meal\", \"rating\", \"date\"])\n",
        "\n",
        "def save_to_history(meal: str, rating: int):\n",
        "    history = load_history()\n",
        "    new_row = pd.DataFrame([{\n",
        "        \"meal\": meal,\n",
        "        \"rating\": rating,\n",
        "        \"date\": pd.Timestamp.now().strftime(\"%Y-%m-%d\")}])\n",
        "    history = pd.concat([history, new_row], ignore_index=True)\n",
        "    history.to_csv(HISTORY_PATH, index=False)\n",
        "    print(f\"Your meal has been eternalised: {meal[:50]}...({rating}/5)\")\n",
        "\n",
        "print(\"Profile functions ready ✅\")\n",
        "print(f\"Existing profile: {load_profile()}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "fcc3d285",
      "metadata": {
        "id": "fcc3d285"
      },
      "source": [
        "## 2. Onboarding UI\n",
        "\n",
        "The first time someone uses the app, they answer a series of questions. This gets saved to their profile CSV and used to build smarter prompts.\n",
        "\n",
        "We're using **Gradio** to build the UI — it creates a web interface directly from Python with no HTML or CSS needed.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "493cd096",
      "metadata": {
        "id": "493cd096",
        "outputId": "301f6d79-72ad-498a-cc29-df1d2050f5cd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 612
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().\n",
            "* Running on public URL: https://206f6dd7e3b3df640e.gradio.live\n",
            "\n",
            "This share link is temporary and will last for up to 1 week (best effort). For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://206f6dd7e3b3df640e.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        }
      ],
      "source": [
        "def save_onboarding(equipment, dietary, allergies, time_available, spice_level, cuisine_prefs):\n",
        "    profile = {\"equipment\": \",\".join(equipment) if equipment else \"hob\",\n",
        "    \"dietary\": \",\".join(dietary) if dietary else \"none\",\n",
        "    \"allergies\": allergies or \"none\",\n",
        "    \"time_available\": time_available,\n",
        "    \"spice_level\": spice_level,\n",
        "    \"cuisine_prefs\": \",\".join(cuisine_prefs) if cuisine_prefs else \"varied\"}\n",
        "    save_profile(profile=profile)\n",
        "    return(f\"Your profile has been eternalised! Equiment: {profile['equipment']}. Time: {profile['time_available']} mins. \")\n",
        "\n",
        "# The Gradio UI is built for you — your job is to wire up the save_onboarding function\n",
        "with gr.Blocks(title=\"AI Cooking Assistant — Setup\") as onboarding_ui:\n",
        "    gr.Markdown(\"## 👋 Let's set up your cooking profile\")\n",
        "    gr.Markdown(\"Answer these once and we'll personalise every suggestion to your kitchen.\")\n",
        "\n",
        "    equipment = gr.CheckboxGroup(\n",
        "        choices=[\"Hob\", \"Oven\", \"Air fryer\", \"Microwave\", \"Instant Pot\", \"BBQ\"],\n",
        "        label=\"What cooking equipment do you have?\",\n",
        "        value=[\"Hob\", \"Oven\"]\n",
        "    )\n",
        "\n",
        "    dietary = gr.CheckboxGroup(\n",
        "        choices=[\"None\", \"Vegetarian\", \"Vegan\", \"Pescatarian\", \"Halal\", \"Kosher\"],\n",
        "        label=\"Dietary requirements\",\n",
        "        value=[\"None\"]\n",
        "    )\n",
        "\n",
        "    allergies = gr.Textbox(\n",
        "        label=\"Any allergies? (e.g. nuts, dairy, gluten)\",\n",
        "        placeholder=\"Leave blank if none\",\n",
        "    )\n",
        "\n",
        "    time_available = gr.Slider(\n",
        "        minimum=10, maximum=120, value=30, step=5,\n",
        "        label=\"How many minutes do you usually have to cook?\"\n",
        "    )\n",
        "\n",
        "    spice_level = gr.Radio(\n",
        "        choices=[\"Mild\", \"Medium\", \"Hot\", \"Extra hot\"],\n",
        "        label=\"Spice preference\",\n",
        "        value=\"Medium\"\n",
        "    )\n",
        "\n",
        "    cuisine_prefs = gr.CheckboxGroup(\n",
        "        choices=[\"Italian\", \"Asian\", \"French\", \"Mexican\", \"Indian\", \"Middle Eastern\", \"British\", \"American\", \"Mediterranean\"],\n",
        "        label=\"Favourite cuisines\",\n",
        "        value=[\"Italian\", \"Asian\"]\n",
        "    )\n",
        "\n",
        "    save_btn = gr.Button(\"Save my profile →\", variant=\"primary\")\n",
        "    output = gr.Textbox(label=\"Status\")\n",
        "\n",
        "    save_btn.click(fn=save_onboarding, inputs=[equipment, dietary, allergies, time_available, spice_level, cuisine_prefs], outputs=output)\n",
        "\n",
        "# share=True is required on Colab\n",
        "onboarding_ui.launch(share=True, debug=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "55e5dd88",
      "metadata": {
        "id": "55e5dd88"
      },
      "source": [
        "## 3. Building the Prompt\n",
        "\n",
        "This is the most interesting part — how you structure the prompt determines the quality of the suggestion.\n",
        "\n",
        "We build a prompt that includes:\n",
        "- The user's full profile\n",
        "- Their past highly-rated meals (so Gemini knows what they like)\n",
        "- Meals they didn't enjoy (so Gemini avoids them)\n",
        "- What they have available tonight\n",
        "\n",
        "**This is prompt engineering** — the context you give the LLM matters as much as the question itself.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "80372bfd",
      "metadata": {
        "id": "80372bfd"
      },
      "outputs": [],
      "source": [
        "def build_prompt(tonight_info: str, profile: dict, history) -> str:\n",
        "    if len(history) > 0:\n",
        "      top_meals = history[history[\"rating\"] >= 4][\"meal\"].tolist()\n",
        "      avoided_meals = history[history[\"rating\"] <= 2][\"meal\"].tolist()\n",
        "    else:\n",
        "      top_meals = {}\n",
        "      avoided_meals = {}\n",
        "\n",
        "    prompt = f\"\"\"You are a personal cooking assistant. Suggest ONE specific meal for tonight based on this persons profile.\n",
        "\n",
        "      THE COOKING SETUP:\n",
        "      - Equipment available: {profile.get('equipment', 'hob')}\n",
        "      - Dietary Requirements:{profile.get('dietary', 'none')}\n",
        "      - Allergies: {profile.get('allergies', 'none')}\n",
        "      - Typical time available: {profile.get('time_available', 30)} minutes\n",
        "      - Spice preference: {profile.get('spice_level', 'medium')}\n",
        "      - Favourite cuisines: {profile.get('cuisine_prefs', 'varied')}\n",
        "\n",
        "      WHAT THEY'VE ENJOYED BEFORE:\n",
        "      {','.join(top_meals) if top_meals else 'None recorded'}\n",
        "\n",
        "      MEALS THEY DIDN'T ENJOY:\n",
        "      {','.join(avoided_meals) if avoided_meals else 'None recorded'}\n",
        "\n",
        "      TONIGHT'S INFO:\n",
        "      {tonight_info}\n",
        "\n",
        "      respond with:\n",
        "      1. the meal name\n",
        "      2. Why it suits them specifically\n",
        "      3  Key ingredients\n",
        "      4. Time to cook\n",
        "      5. How to start\n",
        "\n",
        "    \"\"\"\n",
        "    return prompt\n",
        "# Test it — once you've written the function, run this to see what gets sent to Gemini\n",
        "test_profile = {\n",
        "    \"equipment\": \"Hob, Oven\",\n",
        "    \"dietary\": \"None\",\n",
        "    \"allergies\": \"nuts\",\n",
        "    \"time_available\": 30,\n",
        "    \"spice_level\": \"Medium\",\n",
        "    \"cuisine_prefs\": \"Italian, Asian\"\n",
        "}\n",
        "\n",
        "print(\"=== PROMPT SENT TO GEMINI ===\\n\")\n",
        "print(build_prompt(\n",
        "    \"I have chicken, some pasta, and about 35 minutes\",\n",
        "    test_profile,\n",
        "    load_history()\n",
        "))"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b3a1756d",
      "metadata": {
        "id": "b3a1756d"
      },
      "source": [
        "## 4. The Main App\n",
        "\n",
        "Now we wire everything together — the user describes their situation tonight, we build the prompt from their profile, call Gemini, and show the suggestion.\n",
        "\n",
        "They can then rate it, which gets saved to history and improves future prompts.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "id": "fce0bf5c",
      "metadata": {
        "id": "fce0bf5c",
        "outputId": "36e1c692-7ea1-4db8-b6da-d0a298dd0ee2",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        }
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "module 'google.genai' has no attribute 'GenerativeModel'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_1326/658110968.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Store the last suggestion so we can rate it\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mlast_suggestion\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m\"text\"\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;34m\"\"\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgenai\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mGenerativeModel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"gemini-2.0-flash-lite\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m# TODO: write a get_suggestion(tonight_info) function that:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: module 'google.genai' has no attribute 'GenerativeModel'"
          ]
        }
      ],
      "source": [
        "# Store the last suggestion so we can rate it\n",
        "last_suggestion = {\"text\": \"\"}\n",
        "model = genai.GenerativeModel(\"gemini-2.0-flash-lite\")\n",
        "\n",
        "def get_suggestion(tonight_info: str):\n",
        "    if not tonight_info:\n",
        "      return \"Tell me what you've got ingredients, time and how hungry you are?\", gr.update(visible=False)\n",
        "    profile = load_profile()\n",
        "    if not profile:\n",
        "      return \"Profile not found. Please complete the onboarding first.\", gr.update(visible=False)\n",
        "    history = load_history()\n",
        "    prompt = build_prompt(tonight_info, profile, history)\n",
        "\n",
        "    try:\n",
        "      response = model.generate_content(prompt)\n",
        "      suggestion = response.text\n",
        "      last_suggestion[\"text\"] = suggestion\n",
        "      return suggestion, gr.update(visible=True)\n",
        "    except Exception as e:\n",
        "      return f\"Error with you Gemini pal: {e}\", gr.update(visible=False)\n",
        "\n",
        "def rate_suggestion(rating: int):\n",
        "    if not last_suggestion[\"text\"]:\n",
        "      return \"No suggestion to rate yet\"\n",
        "    save_to_history(last_suggestion[\"text\"][:100], rating)\n",
        "    return f\"Thanks! Rated {rating}/5\"\n",
        "\n",
        "with gr.Blocks(title=\"AI Cooking Assistant\") as main_ui:\n",
        "    gr.Markdown(\"## 🍳 What should I cook tonight?\")\n",
        "\n",
        "    tonight = gr.Textbox(\n",
        "        label=\"What's your situation tonight?\",\n",
        "        placeholder=\"e.g. I've got chicken, some leftover rice, and about 40 minutes. Pretty tired so nothing too complicated.\",\n",
        "        lines=3\n",
        "    )\n",
        "\n",
        "    suggest_btn = gr.Button(\"Suggest something →\", variant=\"primary\")\n",
        "\n",
        "    suggestion_output = gr.Textbox(\n",
        "        label=\"Tonight's suggestion\",\n",
        "        lines=10,\n",
        "        interactive=False\n",
        "    )\n",
        "\n",
        "    with gr.Column(visible=False) as rating_row:\n",
        "        gr.Markdown(\"### How does that sound?\")\n",
        "        with gr.Row():\n",
        "            rating = gr.Slider(minimum=1, maximum=5, value=3, step=1, label=\"Rate this suggestion (1-5)\")\n",
        "            rate_btn = gr.Button(\"Save rating\")\n",
        "        rating_status = gr.Textbox(label=\"\", interactive=False)\n",
        "\n",
        "    suggest_btn.click(\n",
        "        fn=get_suggestion,\n",
        "        inputs=[tonight],\n",
        "        outputs=[suggestion_output, rating_row]\n",
        "        )\n",
        "\n",
        "    rate_btn.click(\n",
        "        fn=rate_suggestion,\n",
        "        inputs=[rating],\n",
        "        outputs=[rating_status]\n",
        "      )\n",
        "\n",
        "# share=True required on Colab\n",
        "main_ui.launch(share=True, debug=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "b2730fab",
      "metadata": {
        "id": "b2730fab"
      },
      "source": [
        "## 5. Your Cooking History\n",
        "\n",
        "See how your profile is building up over time.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "b5d2bcd0",
      "metadata": {
        "id": "b5d2bcd0"
      },
      "outputs": [],
      "source": [
        "print(\"### Your Profile:\")\n",
        "profile = load_profile()\n",
        "if profile:\n",
        "    for key, value in profile.items():\n",
        "        print(f\"- {key.replace('_', ' ').title()}: {value}\")\n",
        "else:\n",
        "    print(\"No profile found. Please complete the onboarding.\")\n",
        "\n",
        "print(\"\\n### Your Cooking History:\")\n",
        "history = load_history()\n",
        "if not history.empty:\n",
        "    display(history)\n",
        "else:\n",
        "    print(\"No cooking history yet.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "045cc4fa",
      "metadata": {
        "id": "045cc4fa"
      },
      "source": [
        "## 6. Mini Challenge (10 mins)\n",
        "\n",
        "Pick one to extend the app:\n",
        "\n",
        "1. **Ingredients tracker** — add a fridge inventory to onboarding, update it after each meal suggestion\n",
        "2. **Weekly planner** — modify the prompt to ask Gemini for 5 meals for the week instead of just tonight\n",
        "3. **Nutritional goal** — add a dietary goal (high protein, low carb etc) to the profile and include it in the prompt\n",
        "4. **Cuisine roulette** — add a \"Surprise me\" button that picks a random cuisine outside their usual preferences\n",
        "\n",
        "Share your Colab link in the chat when you're done! 🚀\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
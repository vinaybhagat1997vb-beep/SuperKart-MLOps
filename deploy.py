
from pathlib import Path
from huggingface_hub import HfApi

# Hugging Face Space repository
SPACE_REPO = "kenshivinay/SuperKart-Sales-App"

# Deployment directory
DEPLOY_DIR = Path(__file__).resolve().parent

# Initialize Hugging Face API
api = HfApi()

# Create Docker Space if it does not exist
api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True
)

print("Hugging Face Space repository is ready!")

# Upload deployment files
api.upload_folder(
    folder_path=str(DEPLOY_DIR),
    repo_id=SPACE_REPO,
    repo_type="space",
    ignore_patterns=[
        "__pycache__",
        "*.pyc",
        ".env",
        ".git/*"
    ],
    commit_message="Deploy SuperKart sales prediction application"
)

print("Deployment files uploaded successfully!")
print(f"https://huggingface.co/spaces/{SPACE_REPO}")

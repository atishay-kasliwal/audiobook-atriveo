# Book to Audio

This repository contains a Google Colab friendly audiobook workflow that uses the official Resemble AI Chatterbox package for multilingual chapter narration. The intended setup is:

1. Edit code locally in VS Code.
2. Push changes to GitHub.
3. Open the notebook in Colab.
4. Pull the latest branch and run generation on a GPU.
5. Save manifests, segment audio, and chapter exports to Google Drive for resume-safe execution.

## Colab Workflow

- Notebook: [notebooks/chatterbox_colab.ipynb](notebooks/chatterbox_colab.ipynb)
- Setup guide: [docs/COLAB_CHATTERBOX_SETUP.md](docs/COLAB_CHATTERBOX_SETUP.md)

The notebook uses the reusable Python modules in `src/audiobook/` instead of embedding all logic directly inside notebook cells.


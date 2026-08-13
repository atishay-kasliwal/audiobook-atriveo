# Colab Chatterbox Setup

This project is designed for a split workflow:

1. Edit the code locally in VS Code.
2. Commit and push your branch to GitHub.
3. Open the notebook in Google Colab.
4. Pull the latest repository state inside Colab.
5. Run generation on a GPU.
6. Save manifests and audio outputs to Google Drive.

The notebook supports both:

- text chapter files such as `.txt`
- scanned or image-based `.pdf` chapter files that need OCR

## Current Repository State

When this integration was added, the workspace did not contain an existing Python package, notebook, text pipeline, dependency file, or git metadata. The current structure was created as a clean additive baseline for the Colab workflow.

## Local VS Code Setup

Use Python 3.11 locally when you want parity with the supported Chatterbox environment.

Recommended commands:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

The core runtime package for Colab lives in `src/audiobook/`. The notebook imports those modules instead of duplicating the pipeline logic inside notebook cells.

## GitHub Synchronization

The notebook does not try to keep a live VS Code session attached to Colab. The reliable bridge is GitHub:

1. Make changes locally in VS Code.
2. Commit the changes.
3. Push the branch to GitHub.
4. Open `notebooks/chatterbox_colab.ipynb` in Colab.
5. Run the repository sync cell to clone or pull the branch.
6. Continue execution on the Colab GPU.

This keeps the development path simple and avoids turning Colab into a persistent remote IDE host.

## Opening The Notebook In Colab

You can open the notebook in either of these ways:

1. Push the repository to GitHub, then in Colab choose `File -> Open notebook -> GitHub` and paste the repository URL.
2. Open the notebook file directly from Google Drive after copying it there.

The notebook expects to clone or update the repository into `/content/workspaces/<repo-name>` before it installs dependencies.

## Selecting A GPU Runtime

Before loading the model:

1. In Colab, choose `Runtime -> Change runtime type`.
2. Set the hardware accelerator to `GPU`.
3. Reconnect the runtime if Colab prompts you.

The notebook prints:

- Python version
- PyTorch version
- CUDA availability
- GPU model
- Available GPU memory
- Chatterbox sample rate after the model loads

If CUDA is unavailable, the notebook stops before chapter generation by default and explains how to select a GPU runtime. CPU fallback exists in the Python package, but the notebook keeps it disabled by default because chapter narration is much slower on CPU.

## Mounting Google Drive

The notebook mounts Google Drive and stores resume-safe outputs under:

- `MyDrive/AudiobookProject/books`
- `MyDrive/AudiobookProject/voices`
- `MyDrive/AudiobookProject/segments`
- `MyDrive/AudiobookProject/chapters`
- `MyDrive/AudiobookProject/manifests`
- `MyDrive/AudiobookProject/logs`

Generated segments and manifests are written immediately so you can resume after a disconnect without rerendering completed work.

## Public Repository Configuration

Set these notebook variables near the top:

- `GITHUB_OWNER`
- `GITHUB_REPO`
- `BRANCH_NAME`
- `PRIVATE_REPOSITORY = False`

The notebook builds `REPOSITORY_URL` automatically from `GITHUB_OWNER` and `GITHUB_REPO`, unless you explicitly set `REPOSITORY_URL_OVERRIDE`.

Then run the repository sync cell. The notebook will clone the repository if it does not exist yet, or fetch and reset the local Colab checkout to the requested branch when it already exists.

## Private Repository Configuration

For private repositories:

1. Set `PRIVATE_REPOSITORY = True`.
2. Run the repository sync cell.
3. Enter a GitHub token when `getpass()` prompts for it.

The notebook uses the token only in memory for that command and avoids printing it into notebook output. Use a token with the smallest repo scope you need.

## Selecting A Narrator Voice

The notebook supports:

1. The default Chatterbox voice with `VOICE_FILE = None`
2. Uploading a narrator clip during the notebook session
3. Selecting a narrator file that already exists in Google Drive
4. Reusing different narrator profiles for different books

Reference clip guidance:

- Prefer a clean 10 to 20 second clip.
- Avoid clipping, heavy background noise, or music.
- Only use voices when you have permission to do so.

The notebook validates file existence, duration, readability, clipping, and supported format before generation begins.

## Choosing The Chapter Input

The notebook now accepts:

1. Plain text chapter files such as `chapter_001.txt`
2. PDF chapter files such as `test1.pdf`

Recommended Drive paths:

- `MyDrive/AudiobookProject/books/<book-id>/chapter_001.txt`
- `MyDrive/AudiobookProject/books/<book-id>/test1.pdf`

If you use a PDF:

- set `CHAPTER_FILE` to the PDF path relative to `OUTPUT_DIRECTORY`, for example `books/demo-book/test1.pdf`
- leave `CHAPTER_INPUT_KIND = "auto"` or set it to `"pdf"`
- keep `FORCE_PDF_OCR = True` for scanned PDFs

The notebook uses `pdftotext` first, then falls back to OCR with `pdftoppm` and `tesseract` when direct extraction is weak.

## Generating A Smoke Test

Before generating a chapter, run the smoke test cell. It synthesizes one short sentence and saves the result as a quick sanity check for:

- dependency installation
- model loading
- language selection
- narrator voice setup

This lets you catch configuration issues before processing a full chapter.

## Generating A Chapter

The chapter flow is:

1. Choose the chapter text file or PDF.
2. If needed, OCR the PDF into chapter text.
3. Clean the text.
4. Preview paragraph-based segments.
5. Prepare or resume the manifest.
6. Generate missing segments with retry handling.
7. Validate the manifest state.
8. Assemble the final chapter audio.
9. Export WAV and optional MP3 or M4B when `ffmpeg` is available.

Important behavior:

- The notebook does not send an entire chapter to Chatterbox in one request.
- Each segment is generated independently.
- Completed segment files are protected from accidental overwrite unless `FORCE_REGENERATION` is explicitly enabled.
- Final chapter export fails clearly if required segments are missing, unless partial export is explicitly enabled.

## Resuming After Disconnection

If Colab disconnects:

1. Reopen the notebook.
2. Mount Drive again.
3. Rerun the repository sync and dependency cells if needed.
4. Rerun the configuration, manifest, and generation cells.

The manifest loader skips segments that are already marked completed and still have matching audio files on Drive. Missing or failed segments are retried without discarding finished work.

## Downloading The Result

The final outputs live in Google Drive under `MyDrive/AudiobookProject/chapters/<book-id>/`.

Typical files:

- `...chapter_001.wav`
- `...chapter_001.mp3`
- `...chapter_001.m4b`

You can download them directly from Drive or keep them there for later batch processing.

## Common CUDA And Dependency Errors

`CUDA is unavailable`

- Confirm that the Colab runtime is using a GPU.
- Reconnect after changing the runtime type.
- Leave `ALLOW_CPU_FALLBACK = False` unless you intentionally want a slow CPU run.

`Torch reports CPU only`

- Restart the runtime and rerun the dependency cell.
- If you changed the install strategy, use the default notebook install path first.

`Voice reference validation failed`

- Confirm the file exists.
- Use a supported format.
- Prefer a clean clip between 10 and 20 seconds.

`Final chapter export refused`

- Open the manifest summary cell and inspect failed or missing segment ids.
- Regenerate those segments before rerunning the assembly cell.

`PDF text extraction looks wrong`

- Keep `FORCE_PDF_OCR = True` for scanned PDFs.
- Lower `MINIMUM_OCR_PAGE_WORDS` if valid low-text pages are being skipped.
- Enable `KEEP_LOW_TEXT_PDF_PAGES = True` if you want every OCR page kept.

## Supported Languages

The current notebook exposes the official open-source Chatterbox multilingual language codes:

`ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh`

These are validated before generation starts.

## Copyright And Voice Permission

Use this workflow only for books, translations, and narrator voices that you are authorized to use. Do not commit or share private voice samples, copyrighted source text without permission, or generated audiobook files that you do not have the right to distribute.

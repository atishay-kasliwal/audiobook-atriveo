from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def build_notebook() -> dict[str, object]:
    cells: list[dict[str, object]] = []

    cells.append(
        markdown_cell(
            """
            # Chatterbox Audiobook Colab Workflow

            This notebook is designed for a VS Code -> GitHub -> Google Colab workflow. Edit locally, push to GitHub, pull the latest branch in Colab, and run multilingual audiobook generation on a GPU while persisting outputs to Google Drive.
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## Workflow Notes

            - Chatterbox performs speech generation only. It does not translate text.
            - The chapter you feed into the model must already be in the target narration language.
            - This notebook supports plain text chapter files and scanned PDFs stored in Google Drive.
            - Completed segment files and manifests are written to Google Drive immediately so the run can resume after disconnects.
            - Use only books and narrator voices that you are authorized to use.
            - If your GitHub repository does not exist yet, publish it from your local repo first, then come back to this notebook.
            """
        )
    )

    cells.append(markdown_cell("## 1. Runtime and GPU verification"))
    cells.append(
        code_cell(
            """
            import platform
            import sys

            IN_COLAB = "google.colab" in sys.modules
            print(f"Running in Colab: {IN_COLAB}")
            print(f"Python version: {platform.python_version()}")

            try:
                import torch
                print(f"PyTorch version: {torch.__version__}")
                print(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    device_index = torch.cuda.current_device()
                    props = torch.cuda.get_device_properties(device_index)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                    print(f"GPU model: {props.name}")
                    print(
                        "Available GPU memory: "
                        f"{free_bytes / 1024**3:.2f} GiB free / {total_bytes / 1024**3:.2f} GiB total"
                    )
                else:
                    print("GPU model: unavailable")
                    print("Available GPU memory: unavailable")
            except Exception as exc:
                print(f"PyTorch check unavailable yet: {exc}")
            """
        )
    )

    cells.append(markdown_cell("## 2. Google Drive mounting"))
    cells.append(
        code_cell(
            """
            from pathlib import Path

            if not IN_COLAB:
                print("Drive mounting is only available inside Colab.")
            else:
                from google.colab import drive
                drive.mount("/content/drive")

            DRIVE_ROOT = Path("/content/drive/MyDrive")
            print(f"Drive root: {DRIVE_ROOT}")
            """
        )
    )

    cells.append(
        markdown_cell(
            """
            ## 3. Configuration

            Update the GitHub settings so Colab can pull your latest VS Code changes. The default chapter file below points at the Drive copy of `test1.pdf` that was uploaded earlier.

            Supported language codes:

            `ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh`
            """
        )
    )
    cells.append(
        code_cell(
            """
            # GitHub synchronization
            GITHUB_OWNER = "atishay-kasliwal"
            GITHUB_REPO = "audiobook-atriveo"
            BRANCH_NAME = "main"
            REPOSITORY_URL_OVERRIDE = None
            PRIVATE_REPOSITORY = False
            REPOSITORY_DIRECTORY_NAME = GITHUB_REPO

            # Official Chatterbox install strategy
            USE_OFFICIAL_GITHUB_SOURCE = False
            OFFICIAL_CHATTERBOX_PACKAGE = "chatterbox-tts==0.1.7"
            OFFICIAL_CHATTERBOX_REF = "master"

            # Runtime and dependency options
            ALLOW_CPU_FALLBACK = False
            PIN_TORCH_TO_OFFICIAL_VERSION = False
            INSTALL_SYSTEM_OCR_TOOLS = True

            # Chapter and output settings
            BOOK_ID = "demo-book"
            CHAPTER_NUMBER = 1
            CHAPTER_FILE = "books/demo-book/test1.pdf"
            CHAPTER_INPUT_KIND = "auto"  # auto, text, pdf
            SOURCE_LANGUAGE = "en"
            TARGET_LANGUAGE = "en"
            VOICE_FILE = None
            OUTPUT_DIRECTORY = "/content/drive/MyDrive/AudiobookProject"

            # PDF extraction controls
            FORCE_PDF_OCR = True
            MINIMUM_DIRECT_PDF_TEXT_ALNUM = 800
            MINIMUM_OCR_PAGE_WORDS = 30
            KEEP_LOW_TEXT_PDF_PAGES = False

            # Segmentation and synthesis
            MINIMUM_CHUNK_CHARACTERS = 200
            MAXIMUM_CHUNK_CHARACTERS = 500
            EXAGGERATION = 0.5
            CFG_WEIGHT = 0.5
            TEMPERATURE = 0.8
            MAXIMUM_RETRIES = 3
            FORCE_REGENERATION = False
            PARTIAL_EXPORT = False
            EXPORT_MP3 = True
            EXPORT_M4B = False
            T3_MODEL = "v3"

            # Optional notebook uploads
            UPLOAD_VOICE_SAMPLE = False
            UPLOAD_CHAPTER_FILE = False

            if REPOSITORY_URL_OVERRIDE:
                REPOSITORY_URL = REPOSITORY_URL_OVERRIDE.strip()
            else:
                REPOSITORY_URL = f"https://github.com/{GITHUB_OWNER.strip()}/{GITHUB_REPO.strip()}.git"

            print("Repository URL:", REPOSITORY_URL)
            print("Branch name:", BRANCH_NAME)
            print("Configured target language:", TARGET_LANGUAGE)
            print("Configured chapter file:", CHAPTER_FILE)
            """
        )
    )

    cells.append(markdown_cell("## 4. GitHub repository cloning or updating"))
    cells.append(
        code_cell(
            """
            import base64
            from getpass import getpass
            import os
            from pathlib import Path
            import subprocess

            WORKSPACE_ROOT = Path("/content/workspaces")
            REPO_PATH = WORKSPACE_ROOT / REPOSITORY_DIRECTORY_NAME

            def ensure_repository_configuration():
                if not REPOSITORY_DIRECTORY_NAME.strip():
                    raise RuntimeError("Set REPOSITORY_DIRECTORY_NAME in the configuration cell.")
                placeholder_values = {"", "YOUR-ACCOUNT", "YOUR-REPO"}
                if REPOSITORY_URL_OVERRIDE is None and (
                    GITHUB_OWNER.strip() in placeholder_values or GITHUB_REPO.strip() in placeholder_values
                ):
                    raise RuntimeError(
                        "Update GITHUB_OWNER and GITHUB_REPO in the configuration cell before running GitHub sync."
                    )
                if not REPOSITORY_URL.startswith("https://github.com/"):
                    raise RuntimeError(
                        "REPOSITORY_URL must point to GitHub. Set REPOSITORY_URL_OVERRIDE or update "
                        "GITHUB_OWNER/GITHUB_REPO first."
                    )

            def git_command(args, *, token=None):
                command = ["git"]
                if token:
                    token_user = "x-access" + "-token"
                    auth = base64.b64encode(f"{token_user}:{token}".encode()).decode()
                    command.extend(
                        [
                            "-c",
                            f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}",
                        ]
                    )
                command.extend(args)
                return command

            def run_git(args, *, token=None):
                print("git " + " ".join(args))
                subprocess.run(git_command(args, token=token), check=True)

            ensure_repository_configuration()

            repo_token = None
            if PRIVATE_REPOSITORY:
                repo_token = getpass("Enter a GitHub token with repository read access: ").strip()

            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            if REPO_PATH.exists():
                print(f"Updating existing repository: {REPO_PATH}")
                remote_result = subprocess.run(
                    ["git", "-C", str(REPO_PATH), "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                )
                if remote_result.returncode == 0 and remote_result.stdout.strip() != REPOSITORY_URL:
                    print(f"Resetting origin from {remote_result.stdout.strip()} to {REPOSITORY_URL}")
                    run_git(["-C", str(REPO_PATH), "remote", "set-url", "origin", REPOSITORY_URL], token=repo_token)

                run_git(["-C", str(REPO_PATH), "fetch", "origin", BRANCH_NAME], token=repo_token)
                branch_check = subprocess.run(
                    ["git", "-C", str(REPO_PATH), "rev-parse", "--verify", BRANCH_NAME],
                    capture_output=True,
                    text=True,
                )
                if branch_check.returncode != 0:
                    run_git(
                        ["-C", str(REPO_PATH), "checkout", "-b", BRANCH_NAME, f"origin/{BRANCH_NAME}"],
                        token=repo_token,
                    )
                else:
                    run_git(["-C", str(REPO_PATH), "checkout", BRANCH_NAME], token=repo_token)
                run_git(["-C", str(REPO_PATH), "pull", "--ff-only", "origin", BRANCH_NAME], token=repo_token)
            else:
                print(f"Cloning repository into: {REPO_PATH}")
                try:
                    run_git(
                        ["clone", "--branch", BRANCH_NAME, "--single-branch", REPOSITORY_URL, str(REPO_PATH)],
                        token=repo_token,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        "GitHub clone failed. Publish the repository from your local machine first, then rerun this cell. "
                        "For example: `gh repo create <repo-name> --source . --private --push`"
                    ) from exc

            repo_token = None
            os.chdir(REPO_PATH)
            print(f"Repository ready at: {REPO_PATH}")
            """
        )
    )

    cells.append(markdown_cell("## 5. Python dependency installation"))
    cells.append(
        code_cell(
            """
            import subprocess
            import sys

            os.chdir(REPO_PATH)

            def pip_install(*args):
                subprocess.run([sys.executable, "-m", "pip", "install", *args], check=True)

            def ensure_import(import_name, *install_args):
                try:
                    __import__(import_name)
                    print(f"Verified Python module: {import_name}")
                except ModuleNotFoundError:
                    if not install_args:
                        raise
                    print(f"Missing Python module '{import_name}'. Installing recovery package...")
                    pip_install(*install_args)
                    __import__(import_name)
                    print(f"Recovered Python module: {import_name}")

            apt_packages = ["ffmpeg"]
            if INSTALL_SYSTEM_OCR_TOOLS or CHAPTER_INPUT_KIND in {"auto", "pdf"}:
                apt_packages.extend(["poppler-utils", "tesseract-ocr"])

            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", *apt_packages], check=True)
            pip_install("--upgrade", "pip", "setuptools", "wheel")

            if PIN_TORCH_TO_OFFICIAL_VERSION:
                pip_install("--upgrade", "torch==2.6.0", "torchaudio==2.6.0")
            else:
                print("Reusing the runtime's existing torch/torchaudio installation.")

            pip_install("-r", "requirements.colab.txt")
            ensure_import(
                "perth",
                "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master",
            )

            if USE_OFFICIAL_GITHUB_SOURCE:
                pip_install(
                    "--no-deps",
                    f"git+https://github.com/resemble-ai/chatterbox.git@{OFFICIAL_CHATTERBOX_REF}",
                )
                print(f"Installed official Chatterbox source from {OFFICIAL_CHATTERBOX_REF}.")
            else:
                pip_install("--no-deps", OFFICIAL_CHATTERBOX_PACKAGE)
                print(f"Installed official Chatterbox package: {OFFICIAL_CHATTERBOX_PACKAGE}")

            ensure_import("perth", "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master")
            ensure_import("chatterbox")
            """
        )
    )

    cells.append(markdown_cell("## 6. Imports and project setup"))
    cells.append(
        code_cell(
            """
            import sys
            from pathlib import Path

            if str(REPO_PATH / "src") not in sys.path:
                sys.path.insert(0, str(REPO_PATH / "src"))

            from audiobook import (
                ChatterboxEngine,
                build_job_config,
                chunk_text,
                export_optional_chapter_formats,
                load_chapter_text,
                prepare_or_resume_manifest,
                render_chapter_audio,
                synthesize_manifest_segments,
            )
            from audiobook.audio_assembler import ffmpeg_available, write_wav
            from audiobook.document_ingestion import SUPPORTED_CHAPTER_SUFFIXES
            from audiobook.text_chunker import normalize_text

            PROJECT_OUTPUT_ROOT = Path(OUTPUT_DIRECTORY).expanduser()
            PROJECT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            OCR_SCRATCH_ROOT = Path("/content/ocr_scratch") / BOOK_ID / f"chapter_{CHAPTER_NUMBER:03d}"

            def resolve_repo_or_output_path(path_value):
                candidate = Path(path_value).expanduser()
                if candidate.is_absolute():
                    return candidate
                repo_candidate = REPO_PATH / candidate
                output_candidate = PROJECT_OUTPUT_ROOT / candidate
                if repo_candidate.exists():
                    return repo_candidate
                return output_candidate

            resolved_chapter_file = resolve_repo_or_output_path(CHAPTER_FILE)
            resolved_voice_file = resolve_repo_or_output_path(VOICE_FILE) if VOICE_FILE else None
            print(f"Initial chapter path: {resolved_chapter_file}")
            print(f"Initial voice path: {resolved_voice_file}")
            print("Supported chapter file formats:", ", ".join(SUPPORTED_CHAPTER_SUFFIXES))
            """
        )
    )

    cells.append(markdown_cell("## 7. Voice sample upload or Drive selection"))
    cells.append(
        code_cell(
            """
            if UPLOAD_VOICE_SAMPLE:
                if not IN_COLAB:
                    raise RuntimeError("Voice upload is only supported inside Colab.")
                from google.colab import files

                uploaded = files.upload()
                if not uploaded:
                    raise RuntimeError("No voice file was uploaded.")
                upload_name, upload_bytes = next(iter(uploaded.items()))
                voice_dir = PROJECT_OUTPUT_ROOT / "voices" / BOOK_ID
                voice_dir.mkdir(parents=True, exist_ok=True)
                uploaded_voice_path = voice_dir / upload_name
                uploaded_voice_path.write_bytes(upload_bytes)
                resolved_voice_file = uploaded_voice_path
                VOICE_FILE = str(uploaded_voice_path)
                print(f"Uploaded voice file: {uploaded_voice_path}")

            voice_validation = None
            if resolved_voice_file:
                validator_engine = ChatterboxEngine(device="cpu", allow_cpu_fallback=True)
                voice_validation = validator_engine.validate_voice_reference(resolved_voice_file)
                print(voice_validation)
            else:
                print("No voice file selected. The default Chatterbox voice will be used.")
            """
        )
    )

    cells.append(markdown_cell("## 8. Text or PDF chapter file selection"))
    cells.append(
        code_cell(
            """
            if UPLOAD_CHAPTER_FILE:
                if not IN_COLAB:
                    raise RuntimeError("Chapter upload is only supported inside Colab.")
                from google.colab import files

                uploaded = files.upload()
                if not uploaded:
                    raise RuntimeError("No chapter file was uploaded.")
                upload_name, upload_bytes = next(iter(uploaded.items()))
                chapter_dir = PROJECT_OUTPUT_ROOT / "books" / BOOK_ID
                chapter_dir.mkdir(parents=True, exist_ok=True)
                uploaded_chapter_path = chapter_dir / upload_name
                uploaded_chapter_path.write_bytes(upload_bytes)
                resolved_chapter_file = uploaded_chapter_path
                CHAPTER_FILE = str(uploaded_chapter_path)
                print(f"Uploaded chapter file: {uploaded_chapter_path}")

            if not resolved_chapter_file.exists():
                raise FileNotFoundError(
                    f"Chapter file not found: {resolved_chapter_file}. Update CHAPTER_FILE or upload a file first."
                )

            suffix = resolved_chapter_file.suffix.lower()
            if suffix not in SUPPORTED_CHAPTER_SUFFIXES:
                raise ValueError(
                    f"Unsupported chapter file format '{suffix}'. Supported formats: {', '.join(SUPPORTED_CHAPTER_SUFFIXES)}"
                )

            if CHAPTER_INPUT_KIND == "auto":
                detected_input_kind = "pdf" if suffix == ".pdf" else "text"
            else:
                detected_input_kind = CHAPTER_INPUT_KIND

            print(f"Using chapter file: {resolved_chapter_file}")
            print(f"Detected input kind: {detected_input_kind}")
            """
        )
    )

    cells.append(markdown_cell("## 9. Build validated configuration"))
    cells.append(
        code_cell(
            """
            job_config = build_job_config(
                repository_url=REPOSITORY_URL,
                branch_name=BRANCH_NAME,
                repository_directory_name=REPOSITORY_DIRECTORY_NAME,
                private_repository=PRIVATE_REPOSITORY,
                book_id=BOOK_ID,
                chapter_number=CHAPTER_NUMBER,
                chapter_file=str(resolved_chapter_file),
                source_language=SOURCE_LANGUAGE,
                target_language=TARGET_LANGUAGE,
                voice_file=str(resolved_voice_file) if resolved_voice_file else None,
                output_directory=OUTPUT_DIRECTORY,
                minimum_chunk_characters=MINIMUM_CHUNK_CHARACTERS,
                maximum_chunk_characters=MAXIMUM_CHUNK_CHARACTERS,
                exaggeration=EXAGGERATION,
                cfg_weight=CFG_WEIGHT,
                temperature=TEMPERATURE,
                maximum_retries=MAXIMUM_RETRIES,
                force_regeneration=FORCE_REGENERATION,
                allow_cpu_fallback=ALLOW_CPU_FALLBACK,
                partial_export=PARTIAL_EXPORT,
                export_mp3=EXPORT_MP3,
                export_m4b=EXPORT_M4B,
                t3_model=T3_MODEL,
            )
            job_config.paths.ensure_directories()
            print(job_config.paths.to_dict())
            """
        )
    )

    cells.append(markdown_cell("## 10. Chapter loading and OCR"))
    cells.append(
        code_cell(
            """
            chapter_load_result = load_chapter_text(
                resolved_chapter_file,
                force_ocr=FORCE_PDF_OCR or detected_input_kind == "pdf",
                minimum_direct_text_alnum=MINIMUM_DIRECT_PDF_TEXT_ALNUM,
                minimum_ocr_page_words=MINIMUM_OCR_PAGE_WORDS,
                keep_low_text_pages=KEEP_LOW_TEXT_PDF_PAGES,
                scratch_directory=OCR_SCRATCH_ROOT,
            )
            raw_chapter_text = chapter_load_result.text

            print(f"Source kind: {chapter_load_result.source_kind}")
            print(f"Used OCR: {chapter_load_result.used_ocr}")
            if chapter_load_result.page_results:
                print(f"Pages kept: {chapter_load_result.kept_page_count}")
                print(f"Pages skipped as low-text: {chapter_load_result.skipped_page_count}")
                for page in chapter_load_result.page_results[:10]:
                    print(
                        f"page={page.page_number}",
                        f"kept={page.kept}",
                        f"words={page.word_count}",
                        f"alnum={page.alnum_count}",
                    )
            print(f"Loaded raw characters: {len(raw_chapter_text)}")
            """
        )
    )

    cells.append(markdown_cell("## 11. Text cleaning"))
    cells.append(
        code_cell(
            """
            cleaned_text = normalize_text(raw_chapter_text)
            print(f"Raw characters: {len(raw_chapter_text)}")
            print(f"Cleaned characters: {len(cleaned_text)}")
            print(cleaned_text[:1200])
            """
        )
    )

    cells.append(markdown_cell("## 12. Paragraph based text segmentation"))
    cells.append(
        code_cell(
            """
            segments_preview = chunk_text(
                cleaned_text,
                chapter_number=CHAPTER_NUMBER,
                language_code=TARGET_LANGUAGE,
                minimum_chunk_characters=MINIMUM_CHUNK_CHARACTERS,
                maximum_chunk_characters=MAXIMUM_CHUNK_CHARACTERS,
            )
            print(f"Total segments: {len(segments_preview)}")
            for segment in segments_preview[:5]:
                print(
                    segment.segment_id,
                    f"paragraph={segment.paragraph_index}",
                    f"chars={segment.char_count}",
                    segment.text[:120],
                )

            manifest, manifest_path = prepare_or_resume_manifest(job_config, chapter_text=cleaned_text)
            print(f"Manifest path: {manifest_path}")
            """
        )
    )

    cells.append(markdown_cell("## 13. Chatterbox model loading"))
    cells.append(
        code_cell(
            """
            import platform
            import torch

            print(f"Python version: {platform.python_version()}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_index)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                print(f"GPU model: {props.name}")
                print(
                    "Available GPU memory: "
                    f"{free_bytes / 1024**3:.2f} GiB free / {total_bytes / 1024**3:.2f} GiB total"
                )
            elif not ALLOW_CPU_FALLBACK:
                raise RuntimeError(
                    "No CUDA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU, "
                    "then rerun this cell. If you intentionally want a slower CPU run, set ALLOW_CPU_FALLBACK = True."
                )

            engine = ChatterboxEngine(t3_model=T3_MODEL, allow_cpu_fallback=ALLOW_CPU_FALLBACK)
            engine.load_model()
            print(f"Chatterbox sample rate: {engine.sample_rate}")
            """
        )
    )

    cells.append(markdown_cell("## 14. Colab smoke test"))
    cells.append(
        code_cell(
            """
            smoke_audio = engine.generate_segment(
                job_config.generation.smoke_test_text,
                language_code=job_config.chapter.target_language,
                audio_prompt_path=str(resolved_voice_file) if resolved_voice_file else None,
                exaggeration=job_config.generation.exaggeration,
                cfg_weight=job_config.generation.cfg_weight,
                temperature=job_config.generation.temperature,
            )
            smoke_path = (
                job_config.paths.chapters_dir
                / BOOK_ID
                / f"{BOOK_ID}_chapter_{CHAPTER_NUMBER:03d}_smoke_test.wav"
            )
            write_wav(smoke_path, smoke_audio, engine.sample_rate)
            print(f"Smoke test written to: {smoke_path}")
            """
        )
    )

    cells.append(markdown_cell("## 15. Audio generation with automatic retry handling"))
    cells.append(
        code_cell(
            """
            manifest = synthesize_manifest_segments(job_config, manifest, engine)
            print("Segment generation pass completed.")
            """
        )
    )

    cells.append(markdown_cell("## 16. Segment validation"))
    cells.append(
        code_cell(
            """
            completed_segments = [segment for segment in manifest.segments if segment.status == "completed"]
            failed_segments = [segment for segment in manifest.segments if segment.status == "failed"]
            pending_segments = [
                segment for segment in manifest.segments if segment.status not in {"completed", "failed"}
            ]

            print(f"Completed segments: {len(completed_segments)}")
            print(f"Failed segments: {len(failed_segments)}")
            print(f"Pending segments: {len(pending_segments)}")

            if failed_segments:
                for segment in failed_segments[:10]:
                    print(segment.id, f"attempts={segment.attempts}", segment.error)
            else:
                print("All required segments are complete.")
            """
        )
    )

    cells.append(markdown_cell("## 17. Chapter audio assembly and WAV export"))
    cells.append(
        code_cell(
            """
            chapter_result = render_chapter_audio(job_config, manifest)
            print(f"WAV chapter export: {chapter_result.chapter_wav_path}")
            """
        )
    )

    cells.append(markdown_cell("## 18. MP3 or M4B export when FFmpeg is available"))
    cells.append(
        code_cell(
            """
            if ffmpeg_available():
                mp3_path, m4b_path = export_optional_chapter_formats(job_config, chapter_result.chapter_wav_path)
                print(f"MP3 export: {mp3_path}")
                print(f"M4B export: {m4b_path}")
            else:
                print("ffmpeg is not available, so optional MP3/M4B exports were skipped.")
            """
        )
    )

    cells.append(markdown_cell("## 19. Saving outputs to Google Drive"))
    cells.append(
        code_cell(
            """
            segment_dir = job_config.paths.segments_dir / BOOK_ID / f"chapter_{CHAPTER_NUMBER:03d}"
            chapter_dir = job_config.paths.chapters_dir / BOOK_ID

            print(f"Manifest: {manifest_path}")
            print(f"Segments directory: {segment_dir}")
            print(f"Chapter directory: {chapter_dir}")
            print("Saved chapter files:")
            for path in sorted(chapter_dir.glob(f"{BOOK_ID}_chapter_{CHAPTER_NUMBER:03d}*")):
                print(f" - {path}")
            """
        )
    )

    cells.append(markdown_cell("## 20. Resume support after interruption"))
    cells.append(
        code_cell(
            """
            resumed_manifest, resumed_manifest_path = prepare_or_resume_manifest(job_config, chapter_text=cleaned_text)
            skipped_on_resume = sum(
                1
                for segment in resumed_manifest.segments
                if segment.status == "completed" and segment.audio_path and Path(segment.audio_path).exists()
            )
            remaining_segment_ids = [
                segment.id for segment in resumed_manifest.segments if segment.status != "completed"
            ]

            print(f"Resume manifest: {resumed_manifest_path}")
            print(f"Segments that will be skipped on resume: {skipped_on_resume}")
            print(f"Segments still needing work: {remaining_segment_ids}")
            """
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "chatterbox_colab.ipynb"
    notebook_path.write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    main()

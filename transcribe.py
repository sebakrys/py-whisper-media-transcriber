import argparse
import math
from pathlib import Path
from time import perf_counter
from datetime import datetime, timedelta

import torch
import whisper


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".webm",
}


def is_media_file(path: Path) -> bool:
    """Sprawdza czy plik wygląda na wideo/audio po rozszerzeniu."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def parse_datetime_from_stem(stem: str) -> datetime | None:
    """
    Próbuje sparsować datę/czas z nazwy pliku bez rozszerzenia.
    Obsługuje np.:
    - '2025-11-26 16-44-59'
    - '2025-11-26_16-44-59'
    """
    for fmt in ("%Y-%m-%d %H-%M-%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return None


def build_lines_from_segments(segments, pause_threshold: float = 2.0) -> str:
    """
    Buduje tekst z segmentów Whispera, robiąc nową linię
    przy pauzie dłuższej niż pause_threshold (w sekundach).
    """
    if not segments:
        return ""

    lines: list[str] = []
    current_line = segments[0]["text"].strip()
    prev_end = segments[0]["end"]

    for seg in segments[1:]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        gap = start - prev_end
        if gap >= pause_threshold:
            # nowy akapit / linia
            if current_line.strip():
                lines.append(current_line.strip())
            current_line = text
        else:
            # kontynuacja tej samej linii
            if text:
                if current_line:
                    current_line += " " + text
                else:
                    current_line = text

        prev_end = end

    if current_line.strip():
        lines.append(current_line.strip())

    return "\n".join(lines)


def transcribe_file(
    input_path: Path,
    model,
    device: str = "cpu",
    language: str = "pl",
    pause_threshold: float = 2.0,
) -> tuple[str, float]:
    """
    Transkrybuje JEDEN plik za pomocą już załadowanego modelu.
    Zwraca (tekst, czas_trwania_w_sekundach).
    """
    print(f"[INFO] Ładuję audio z pliku: {input_path}")
    audio = whisper.load_audio(str(input_path))
    sr = whisper.audio.SAMPLE_RATE  # standardowo 16000 Hz

    total_duration_sec = len(audio) / sr
    total_duration_min = total_duration_sec / 60
    print(f"[INFO] Długość nagrania: {total_duration_min:.1f} min ({total_duration_sec:.1f} s)")

    print("[INFO] Transkrypcja całego pliku (Whisper sam dzieli na segmenty)...")
    t0 = perf_counter()
    result = model.transcribe(
        audio=audio,
        language=language,
        task="transcribe",
        fp16=(device == "cuda"),
        verbose=True,  # pokazuje segmenty + tekst w konsoli
    )
    dt = perf_counter() - t0
    print(f"[INFO] Transkrypcja pliku zakończona w {dt:.1f} s")

    segments = result.get("segments", [])
    text = build_lines_from_segments(segments, pause_threshold=pause_threshold)

    return text, total_duration_sec


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transkrypcja pliku lub katalogu z plikami wideo na tekst za pomocą Whisper (CPU/GPU). "
            "Dla katalogu wszystkie pliki wideo są sortowane rosnąco po nazwie i łączone w jeden plik txt. "
            "Nowe linie są wstawiane przy dłuższych pauzach w mowie."
        )
    )
    parser.add_argument(
        "input",
        help="Ścieżka do pliku wideo (np. film.mp4) lub katalogu z plikami wideo",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Plik wyjściowy .txt (dla katalogu: jeden wspólny plik). "
            "Domyślnie: dla pliku -> nazwa_pliku.txt, "
            "dla katalogu -> nazwa zależna od pierwszego pliku i czasu trwania całości."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help="Model Whisper (tiny, base, small, medium, large). Domyślnie: small",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="pl",
        help="Kod języka mowy (np. pl, en). Domyślnie: pl",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda"],
        help="Wymuś urządzenie: cpu lub cuda (domyślnie auto).",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=2.0,
        help="Przerwa (w sekundach) po której zaczynany jest nowy wiersz. Domyślnie 2.0.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Ścieżka nie istnieje: {input_path}")

    # Ustalenie listy plików do transkrypcji
    if input_path.is_file():
        if not is_media_file(input_path):
            raise ValueError(f"Plik nie wygląda na wideo/audio: {input_path}")
        files_to_process = [input_path]
        is_directory_mode = False
    elif input_path.is_dir():
        files_to_process = sorted(
            [p for p in input_path.iterdir() if is_media_file(p)],
            key=lambda p: p.name.lower(),
        )
        is_directory_mode = True
        if not files_to_process:
            raise ValueError(f"W katalogu {input_path} nie znaleziono żadnych plików wideo.")
    else:
        raise ValueError(f"Ścieżka nie jest ani plikiem, ani katalogiem: {input_path}")

    # Wybór urządzenia
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] Globalne urządzenie: {device}")
    print(f"[INFO] Ładuję model Whisper: {args.model} (tylko raz)...")
    model = whisper.load_model(args.model, device=device)

    # Ustalenie ścieżki wyjściowej (jeden wspólny plik)
    # Nazwa zależna od pierwszego pliku i długości całości
    if args.output:
        output_path = Path(args.output)
    else:
        if is_directory_mode:
            first_stem = files_to_process[0].stem
            last_stem = files_to_process[-1].stem
            # na razie tylko placeholder, właściwą nazwę ustalimy po poznaniu długości
            output_path = input_path / "transcription_all_tmp.txt"
        else:
            output_path = input_path.with_suffix(".txt")

    print(f"[INFO] Liczba plików do przetworzenia: {len(files_to_process)}")
    print(f"[INFO] Wstępny plik wyjściowy: {output_path}")

    all_text_parts: list[str] = []
    durations_sec: list[float] = []

    for idx, file_path in enumerate(files_to_process, start=1):
        print("\n" + "=" * 80)
        print(f"[INFO] Plik {idx}/{len(files_to_process)}: {file_path.name}")
        print("=" * 80)

        text, duration_sec = transcribe_file(
            input_path=file_path,
            model=model,
            device=device,
            language=args.language,
            pause_threshold=args.pause_threshold,
        )

        durations_sec.append(duration_sec)

        header = f"\n\n===== PLIK: {file_path.name} =====\n\n"
        all_text_parts.append(header + text.strip())

    final_text = "".join(all_text_parts).strip() + "\n"

    # Jeśli pracujemy na katalogu i nie podano -o, ustalamy lepszą nazwę
    if is_directory_mode and not args.output:
        first_stem = files_to_process[0].stem
        last_stem = files_to_process[-1].stem

        total_duration_sec = sum(durations_sec)
        total_hours = int(total_duration_sec // 3600)
        total_minutes = int((total_duration_sec % 3600) // 60)

        start_dt = parse_datetime_from_stem(first_stem)
        last_start_dt = parse_datetime_from_stem(last_stem)
        end_dt: datetime | None = None

        if last_start_dt is not None:
            end_dt = last_start_dt + timedelta(seconds=durations_sec[-1])

        if start_dt is not None and end_dt is not None:
            # nazwa: start__koniec
            name = (
                f"{start_dt.strftime('%Y-%m-%d %H-%M-%S')}"
                f"__{end_dt.strftime('%Y-%m-%d %H-%M-%S')}_medium.txt"
            )
        else:
            # fallback: pierwszy plik + długość całości
            name = f"{first_stem}_len_{total_hours:02d}h{total_minutes:02d}m_medium.txt"

        output_path = input_path / name
        print(f"[INFO] Ostateczna nazwa pliku wyjściowego: {output_path}")

    output_path.write_text(final_text, encoding="utf-8")
    print("\n[INFO] Zapisano transkrypcję do:", output_path)


if __name__ == "__main__":
    main()

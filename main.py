"""Run the deterministic mini-RAG pipeline."""

from pathlib import Path

from rag_pipeline import Pipeline, PipelineError


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        pipeline = Pipeline(root)
        pipeline.run()
    except (OSError, ValueError, PipelineError) as exc:
        print(f"Pipeline failed: {exc}")
        return 1

    print(f"Pipeline complete: {pipeline.state.value}")
    print(f"Artifacts written to {root / 'artifacts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

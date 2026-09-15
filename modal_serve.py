"""Modal deployment wrapper around the FastAPI app in serve.py.

Deploy with: uv run modal deploy modal_serve.py
"""

import modal

# Must match the sklearn_version recorded in pipeline.joblib's metadata
# (see build_pipeline.py / bundle["metadata"]["sklearn_version"]) - unpickling
# a scikit-learn estimator with a mismatched version is a common source of
# silent breakage.
SKLEARN_VERSION = "1.9.1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi",
        "pydantic",
        "pandas",
        "numpy",
        "joblib",
        f"scikit-learn=={SKLEARN_VERSION}",
    )
    # copy=True bakes these three files into the image layer at build time,
    # rather than mounting them in at container startup.
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py", copy=True)
    .add_local_file("serve.py", "/root/serve.py", copy=True)
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib", copy=True)
)

app = modal.App("nl-mvp-pipeline", image=image)


@app.function()
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from serve import app as web_app  # imported inside the function, not at module import time

    return web_app

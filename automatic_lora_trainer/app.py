import uvicorn

from .api import create_app
from .settings import AI_TOOLKIT_DIR, HOST, PORT, WORK_ROOT


def main():
    print(f"Work root: {WORK_ROOT}")
    print(f"ai-toolkit: {AI_TOOLKIT_DIR}")
    try:
        from google.colab import output

        print("Colab proxy URL:")
        print(output.eval_js(f"google.colab.kernel.proxyPort({PORT})"))
    except Exception:
        pass
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()

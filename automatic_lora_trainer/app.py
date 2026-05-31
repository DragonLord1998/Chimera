from .settings import AI_TOOLKIT_DIR, HOST, PORT, SHARE, WORK_ROOT
from .ui import build_app


def main():
    print(f"Work root: {WORK_ROOT}")
    print(f"ai-toolkit: {AI_TOOLKIT_DIR}")
    try:
        from google.colab import output

        print("Colab proxy URL:")
        print(output.eval_js(f"google.colab.kernel.proxyPort({PORT})"))
    except Exception:
        pass
    build_app().queue().launch(server_name=HOST, server_port=PORT, share=SHARE)


if __name__ == "__main__":
    main()

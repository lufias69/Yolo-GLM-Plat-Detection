#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def _ensure_research_env():
    """The webapp needs torch/ultralytics/easyocr, which live only in the
    research environment (mace_md). If started with the wrong interpreter,
    transparently re-run this same command with the right one."""
    import importlib.util
    if importlib.util.find_spec('ultralytics'):
        return
    candidates = [os.environ.get('ANPR_PYTHON'),
                  r'C:\Users\lufias\miniforge3\envs\mace_md\python.exe']
    me = os.path.realpath(sys.executable)
    for c in candidates:
        if c and os.path.exists(c) and os.path.realpath(c) != me:
            print(f'[manage.py] interpreter {sys.executable} tidak punya ultralytics; '
                  f'jalan ulang dengan {c}', file=sys.stderr)
            os.execv(c, [c, os.path.realpath(__file__), *sys.argv[1:]])
    raise ImportError(
        'Webapp butuh environment riset (torch, ultralytics, easyocr). '
        'Jalankan dengan: C:\\Users\\lufias\\miniforge3\\envs\\mace_md\\python.exe '
        'manage.py <perintah>  — atau set env var ANPR_PYTHON ke interpreter yang benar.')


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anpr_web.settings')
    _ensure_research_env()
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

import sys

from django.apps import AppConfig


class TestingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'testing'

    def ready(self):
        # Auto-start the worker only for a real server run: not tests, not mgmt commands
        # other than runserver, and not the autoreloader's parent process.
        import os
        if os.environ.get('MANAGE_RUNTESTS'):
            return
        if 'test' in sys.argv:
            return
        if 'runserver' in sys.argv:
            from testing.services.workers import start_worker
            start_worker()

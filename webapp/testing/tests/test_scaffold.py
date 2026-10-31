from django.test import Client
from django.test import TestCase


class ScaffoldSmokeTest(TestCase):
    def test_home_renders_anpr_title(self):
        response = Client().get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('ANPR', response.content.decode())

"""Bounded offline dependency regressions; no CUDA, checkpoints, or HTTP.

Run with ``python -m unittest discover -s tests -v`` after installing the two
requirements and Requests' dependencies.  This is not an inference test.
"""
import ast
import io
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

import requests
from requests.adapters import BaseAdapter, HTTPAdapter
from requests.cookies import cookiejar_from_dict
from tqdm import tqdm
from tqdm.cli import cast
from tqdm.std import TqdmTypeError


ROOT = Path(__file__).resolve().parents[1]


class SyntheticAdapter(BaseAdapter):
    """Serve only a tiny in-memory response, never a network request."""

    def __init__(self, confirmation=False):
        self.confirmation = confirmation
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request.url, kwargs))
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.url = request.url
        response.raw = io.BytesIO(b'synthetic-payload')
        response.cookies = cookiejar_from_dict(
            {'download_warning_fixture': 'fixture-token'}
            if self.confirmation and len(self.calls) == 1 else {})
        return response

    def close(self):
        pass


class DependencyCompatibilityTests(unittest.TestCase):
    def setUp(self):
        # Fail closed if a code change accidentally escapes our adapter.
        for name in ('create_connection', 'getaddrinfo', 'socket'):
            guard = mock.patch.object(
                socket, name, side_effect=AssertionError('network forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_expected_pins(self):
        requirements = (ROOT / 'requirements.txt').read_text().splitlines()
        self.assertIn('requests==2.32.4', requirements)
        self.assertIn('tqdm==4.66.3', requirements)
        self.assertEqual(requests.__version__, '2.32.4')

    def test_application_streaming_and_confirmation(self):
        # Compile only these real pure helpers; do not import the torch module.
        tree = ast.parse((ROOT / 'src/imaginaire/utils/io.py').read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('download_file', 'get_confirm_token',
                                      'save_response_content')]
        self.assertEqual(len(selected), 3)
        module = ast.Module(body=selected, type_ignores=[])
        namespace = {'requests': requests}
        exec(compile(module, 'src/imaginaire/utils/io.py', 'exec'), namespace)
        for confirmation in (False, True):
            with self.subTest(confirmation=confirmation):
                adapter = SyntheticAdapter(confirmation)
                session = requests.Session()
                # Never read user .netrc/proxy configuration in a test.
                session.trust_env = False
                session.mount('https://', adapter)
                self.addCleanup(session.close)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'tiny-fixture.bin'
                    with mock.patch.object(requests, 'Session', return_value=session):
                        namespace['download_file']('https://fixture.invalid/file', output)
                    self.assertEqual(output.read_bytes(), b'synthetic-payload')
                self.assertEqual(len(adapter.calls), 2 if confirmation else 1)
                self.assertTrue(all(kwargs['stream'] for _, kwargs in adapter.calls))
                self.assertTrue(all(kwargs['verify'] for _, kwargs in adapter.calls))
                if confirmation:
                    self.assertIn('confirm=fixture-token', adapter.calls[-1][0])

    def test_netrc_lookup_uses_hostname_not_userinfo(self):
        fake = mock.Mock()
        fake.authenticators.return_value = None
        # The path is synthetic and parsing is mocked: no real credential read.
        with mock.patch.dict(os.environ, {'NETRC': '/synthetic/never-read.netrc'}), \
                mock.patch('requests.utils.os.path.exists', return_value=True), \
                mock.patch('netrc.netrc', return_value=fake):
            requests.utils.get_netrc_auth('http://victim.invalid:80@attacker.invalid/')
        fake.authenticators.assert_called_once_with('attacker.invalid')

    def test_https_proxy_authorization_not_forwarded(self):
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        prepared = requests.Request('GET', 'https://fixture.invalid/').prepare()
        prepared.headers['Proxy-Authorization'] = 'synthetic-do-not-forward'
        session.rebuild_proxies(prepared, {'https': 'http://fixture:fixture@proxy.invalid:8080'})
        self.assertNotIn('Proxy-Authorization', prepared.headers)

    def test_tls_pool_parameters_separate_verify_modes(self):
        adapter = HTTPAdapter()
        self.addCleanup(adapter.close)
        prepared = requests.Request('GET', 'https://fixture.invalid/').prepare()
        _, unverified = adapter.build_connection_pool_key_attributes(prepared, False)
        _, verified = adapter.build_connection_pool_key_attributes(prepared, True)
        self.assertNotEqual(unverified, verified)
        self.assertEqual(unverified['cert_reqs'], 'CERT_NONE')
        self.assertEqual(verified['cert_reqs'], 'CERT_REQUIRED')

    def test_tqdm_iterable_total_and_description(self):
        output = io.StringIO()
        with tqdm(range(3), total=3, file=output, mininterval=0,
                  desc='fixture', ncols=40) as progress:
            self.assertEqual(list(progress), [0, 1, 2])
            self.assertEqual(progress.n, 3)
        self.assertIn('fixture', output.getvalue())

    def test_tqdm_cli_rejects_expression_without_executing(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'must-not-exist'
            expression = '__import__("pathlib").Path(%r).touch()' % str(sentinel)
            self.assertEqual(cast(expression, 'str'), expression)
            with self.assertRaises(TqdmTypeError):
                cast(expression, 'int')
            self.assertFalse(sentinel.exists())


if __name__ == '__main__':
    unittest.main()

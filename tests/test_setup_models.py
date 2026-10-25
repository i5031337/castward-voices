"""Exercise model selection, portable configs, and install failures without downloads."""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from backend_config import ServerConfig
from launch_config import prepare_configs
import setup_models as setup


class SetupModelTests(unittest.TestCase):
    def setUp(self):
        self.catalog = setup.read_json(setup.ROOT / 'model_catalog.json')

    def test_each_selection_produces_valid_configs_and_dependencies(self):
        names = list(self.catalog['models'])
        for length in range(1, len(names) + 1):
            for selection in itertools.combinations(names, length):
                with self.subTest(selection=selection):
                    native, speech = setup.make_configs(self.catalog, selection)
                    config = ServerConfig.model_validate(speech)
                    ids = {model['id'] for model in native['models']}
                    for profile in config.models.values():
                        for key in ('backend_model', 'clone_model', 'design_model', 'streaming_model'):
                            target = getattr(profile, key)
                            if target:
                                self.assertIn(target, ids)
                    for model in native['models']:
                        self.assertTrue(Path(model['path']).is_absolute())
                        self.assertEqual(model['device'], 0)
                    self.assertEqual(set(speech['models']), set().union(*(
                        set(self.catalog['models'][name]['profiles']) for name in selection)))

    def test_selection_add_replace_all_and_unknown(self):
        previous = {'models': ['ming', 'kokoro']}
        self.assertEqual(setup.select_models(self.catalog, previous, add='pocket,ming'),
                         ['ming', 'kokoro', 'pocket'])
        self.assertEqual(setup.select_models(self.catalog, previous, models='pocket'), ['pocket'])
        self.assertEqual(setup.select_models(self.catalog, previous, models='all'), list(self.catalog['models']))
        with self.assertRaisesRegex(ValueError, 'Unknown models'):
            setup.select_models(self.catalog, previous, add='missing')

    def test_future_model_uses_catalog_without_installer_branches(self):
        import copy
        catalog = copy.deepcopy(self.catalog)
        catalog['models']['future'] = {
            **catalog['models']['pocket'], 'family': 'future_tts', 'profiles': ['future'],
            'weights': {'future-native': 'future/model.gguf'}, 'include': ['future/*']}
        native = setup.read_json(setup.ROOT / 'audiocpp-server.local.json')
        speech = setup.read_json(setup.ROOT / 'backend.json')
        native['models'].append({'id': 'future-native', 'family': 'future_tts', 'path': 'old.gguf'})
        speech['models']['future'] = {'backend_model': 'future-native'}
        with patch.object(setup, 'read_json', side_effect=[native, speech]):
            generated, public = setup.make_configs(catalog, ['future'])
        self.assertEqual([model['id'] for model in generated['models']], ['future-native'])
        self.assertEqual(set(public['models']), {'future'})
        ServerConfig.model_validate(public)
        with tempfile.TemporaryDirectory() as directory:
            configure, _ = setup.build_commands(setup.ROOT, Path(directory), catalog, cpu=True, jobs=2)
        self.assertTrue(any(arg.endswith(',future_tts') for arg in configure))

    def test_cpu_config_and_catalog_cpu_routing(self):
        native, speech = setup.make_configs(self.catalog, ['ming', 'kokoro', 'pocket'], cpu=True)
        self.assertEqual(native['backend'], 'cpu')
        self.assertTrue(all('device' not in model for model in native['models']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, data in [('native.json', native), ('backend.json', speech)]:
                (root / filename).write_text(json.dumps(data))
            prepare_configs(root / 'native.json', root / 'backend.json', root / 'split')
            routed = setup.read_json(root / 'split/backend.json')['backend_routes']
            self.assertEqual(set(routed), {'kokoro', 'kokoro-stream', 'pocket-english'})

    def test_install_only_commits_selection_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory) / 'install'
            environment = {'BUILD_JOBS': '2', 'AUDIOCPP_SOURCE': str(setup.ROOT.parent / 'Ming/audio.cpp'),
                           'AUDIOCPP_BUILD_DIR': str(Path(directory) / 'build')}
            def run(args, failure=None):
                with patch.object(setup, 'INSTALL', install), patch.dict(os.environ, environment, clear=True), \
                        patch('sys.argv', ['setup_models.py', *args]), \
                        patch.object(setup.subprocess, 'run', side_effect=failure) as calls, \
                        contextlib.redirect_stdout(io.StringIO()):
                    setup.main()
                return calls
            calls = run(['--models', 'pocket', '--cpu'])
            self.assertEqual(setup.read_json(install / 'settings.json')['models'], ['pocket'])
            self.assertEqual(set(setup.read_json(install / 'backend.json')['models']), {'pocket-tts'})
            download = next(call.args[0] for call in calls.call_args_list if 'download' in call.args[0])
            self.assertIn('PocketTTS-GGUF/english/*', download)
            before = {path.name: path.read_bytes() for path in install.iterdir()}
            with self.assertRaises(SystemExit):
                run(['--add', 'firered'], subprocess.CalledProcessError(1, ['cmake']))
            self.assertEqual(before, {path.name: path.read_bytes() for path in install.iterdir()})
            run(['--add', 'firered', '--plan'])
            self.assertEqual(before, {path.name: path.read_bytes() for path in install.iterdir()})


if __name__ == '__main__':
    unittest.main()

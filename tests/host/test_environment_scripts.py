"""Run with python3 -m unittest discover -s tests/host; no container or GPU needed."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        shutil.copytree(str(ROOT / 'tools'), str(self.root / 'tools'),
                        ignore=shutil.ignore_patterns('site.local.sh', '__pycache__'))
        for name in ('activate.sh', 'install.sh'):
            shutil.copy(str(ROOT / name), str(self.root / name))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(
            ('LUMI_', 'JAX', 'VENV', 'SIF', 'IMAGES_DIR', 'LAIF_', 'LD_', 'PYTHONPATH', 'BASH_ENV'))}
        self.env['LUMI_JAX_WORK'] = str(Path(self.temp.name) / 'work')

    def run_shell(self, command):
        return subprocess.check_output(['bash', '-eu', '-c', command],
                                       cwd=str(self.root), env=self.env,
                                       universal_newlines=True).strip()

    def test_install_products_and_override(self):
        (self.root / 'tools/install_layered_venv.sh').write_text(
            '#!/bin/bash\nmkdir -p "$2"\nprintf "%s\\n" "$1" "$3" > "$2/arguments"\n')
        image = Path(self.temp.name) / 'image.sif'
        image.touch()
        self.env['SIF'] = str(image)
        for product, script, line in [('lean', 'install.sh', '0.11.2')]:
            for override in (False, True):
                with self.subTest(product=product, override=override):
                    dest = Path(self.env['LUMI_JAX_WORK']) / (
                        'custom-' + product if override else 'jax-venv')
                    if override:
                        self.env['VENV_DIR'] = str(dest)
                        self.env['LUMI_JAX_LINE'] = '0.11.0'
                    else:
                        self.env.pop('VENV_DIR', None)
                        self.env.pop('LUMI_JAX_LINE', None)
                    self.run_shell('module() { :; }; export -f module; bash ' + script)
                    self.assertEqual((dest / 'arguments').read_text().splitlines(),
                                     [str(image), '0.11.0' if override else line])
                    self.assertEqual((dest / 'lumi-jax-variant').read_text().strip(), product)

    def test_activation_override_and_native_path_cleanup(self):
        dest = Path(self.temp.name) / 'custom'
        (dest / 'bin').mkdir(parents=True)
        (dest / 'bin/activate').write_text('export VIRTUAL_ENV="$VENV_DIR"\n')
        (dest / 'lumi-jax-native-rocm').touch()
        self.env['VENV_DIR'] = str(dest)
        for script in ('activate.sh',):
            out = self.run_shell('export LD_LIBRARY_PATH="$PWD/shim:/keep"; '
                                 'export PYTHONPATH="/opt/lumi-jax:/keep/python"; '
                                 'source ' + script + '; '
                                 'printf "%s\\n" "$VIRTUAL_ENV" "$LD_LIBRARY_PATH" "$PYTHONPATH"')
            self.assertEqual(out.splitlines(), [str(dest), '/keep', '/keep/python'])

    def test_recorded_image_survives_latest_change(self):
        dest = Path(self.env['LUMI_JAX_WORK']) / 'jax-venv'
        dest.mkdir(parents=True)
        old = dest / 'immutable.sif'
        old.touch()
        for name, value in [('line', '0.11.2'), ('variant', 'lean'), ('image', str(old))]:
            (dest / ('lumi-jax-' + name)).write_text(value + '\n')
        self.assertEqual(self.run_shell('source tools/resolve_image.sh; '
                                       'jax_resolve_image lean'), str(old))
        old.unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_shell('source tools/resolve_image.sh; jax_resolve_image lean')

    def test_removed_product_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_shell('source tools/resolve_image.sh; jax_resolve_image torch')

    def test_build_setup_override(self):
        scratch = Path(self.temp.name) / 'build'
        self.env['BUILD_SCRATCH'] = str(scratch)
        out = self.run_shell('REPO_DIR="$PWD"; WORK="$LUMI_JAX_WORK"; '
                             'IMAGES_DIR="$WORK/images"; proot() { :; }; '
                             'singularity() { :; }; source tools/build_common.sh >/dev/null; '
                             'test -d "$IMAGES_DIR"; test -d "$SINGULARITY_CACHEDIR"; '
                             'printf "%s\\n" "$SINGULARITY_TMPDIR" "$TMPDIR" "$PWD"')
        self.assertEqual(out.splitlines(), [str(scratch / 'sing-tmp')] * 2 + [str(self.root)])

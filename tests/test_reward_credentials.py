import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reward_calculation as rc


class JudgeCredentialFileTests(unittest.TestCase):
    def secret_file(self, value='secret-value', mode=stat.S_IRUSR | stat.S_IWUSR):
        root=tempfile.TemporaryDirectory()
        path=Path(root.name)/'deepinfra.key'
        path.write_text(value)
        path.chmod(mode)
        self.addCleanup(root.cleanup)
        return path

    def test_config_reads_key_file_path_without_exposing_it_in_summary(self):
        path=self.secret_file()
        with patch.dict(os.environ,{'DEEPINFRA_API_KEY_FILE':str(path)},clear=False):
            config=rc.RewardConfig.from_env()
        self.assertEqual(getattr(config,'api_key_file',None),str(path))
        self.assertNotIn(str(path),repr(config.summary()))

    def test_secure_key_file_is_loaded_transiently(self):
        path=self.secret_file('  secret-value\n')
        config=rc.RewardConfig(judge_backend='deepinfra',api_key_file=str(path))
        async def load():
            judge=rc.DeepSeekJudge(config)
            loader=getattr(judge,'_load_api_key',None)
            self.assertTrue(callable(loader))
            value=await loader() if loader else ''
            self.assertEqual(judge.api_key,'')
            return value
        self.assertEqual(asyncio.run(load()),'secret-value')

    def test_group_readable_key_file_is_rejected(self):
        path=self.secret_file(mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
        config=rc.RewardConfig(judge_backend='deepinfra',api_key_file=str(path))
        async def load():
            judge=rc.DeepSeekJudge(config)
            loader=getattr(judge,'_load_api_key',None)
            self.assertTrue(callable(loader))
            if loader: return await loader()
        if getattr(rc.DeepSeekJudge,'_load_api_key',None):
            with self.assertRaisesRegex(RuntimeError,'permissions'):
                asyncio.run(load())

    def test_empty_key_file_is_rejected(self):
        path=self.secret_file(' \n')
        config=rc.RewardConfig(judge_backend='deepinfra',api_key_file=str(path))
        async def load():
            judge=rc.DeepSeekJudge(config)
            loader=getattr(judge,'_load_api_key',None)
            self.assertTrue(callable(loader))
            if loader: return await loader()
        if getattr(rc.DeepSeekJudge,'_load_api_key',None):
            with self.assertRaisesRegex(RuntimeError,'empty'):
                asyncio.run(load())


if __name__=='__main__':
    unittest.main()

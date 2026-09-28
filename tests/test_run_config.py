import unittest
from pathlib import Path
import json

class RunConfigTests(unittest.TestCase):
    def test_final_config_requires_selected_lr_and_seed(self):
        import run_workshop
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/nemotron35_lightning_grpo_repaired_final.json').read_text())
        with self.assertRaises(ValueError):run_workshop.config_environment(config)
        result=run_workshop.config_environment(config,lr=0.00001,seed=0)
        self.assertEqual(result['EPOCHS'],'3')
        self.assertEqual(result['STEPS'],'36')
        self.assertEqual(result['DEV_SPLIT'],'dev')
        self.assertEqual(result['REMOVE_CONSTANT_REWARD_GROUPS'],'true')

    def test_reject_eval_selection(self):
        import run_workshop
        with self.assertRaises(ValueError):
            run_workshop.config_environment({'learning_rate':1e-5,'seed':0,'eval_split':'eval'})

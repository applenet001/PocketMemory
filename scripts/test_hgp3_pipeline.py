"""Offline checks for review gates, provenance and user/text leakage. No API calls."""
import tempfile
import unittest
from pathlib import Path
from hgp3_data import LABELS, digest, load_training, read, write
from label_hgp3 import review, finalize


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.rows = []
        for i in range(3000):
            split = 'train' if i < 2400 else 'val' if i < 2700 else 'test'
            source = 'MobileMem' if i < 600 else 'fixture'
            self.rows.append(dict(id=str(i), text=f'unique evidence {i}', context='',
                group_id=str(i), source=source, split=split, label=LABELS[i % 3],
                uncertain=False, reviewed=False, annotation_source='fixture'))
        write(self.out / 'candidates.jsonl', self.rows)
        write(self.out / 'teacher.jsonl', [dict(id=r['id'], label=r['label'], uncertain=False,
            reason='fixture only', text_sha256=digest(r['text']), annotation_source='deepseek', reviewed=False)
            for r in self.rows[:600]])

    def tearDown(self):
        self.temp.cleanup()

    def test_review_and_finalize_gate(self):
        review(self.out, .2, 42)
        queue = read(self.out / 'review.jsonl')
        self.assertEqual(len(queue), 120)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            finalize(self.out)
        for r in queue:
            r.update(final_label=r['label'], reviewed=True, reviewer='unit-test-fixture')
        queue[0]['final_label'] = LABELS[(LABELS.index(queue[0]['label']) + 1) % 3]
        write(self.out / 'review.jsonl', queue)
        finalize(self.out)
        rows = load_training(self.out / 'data.jsonl')
        result = {r['id']: r for r in rows}[queue[0]['id']]
        self.assertEqual(result['label'], queue[0]['final_label'])
        self.assertEqual(result['annotation_source'], 'human')

    def test_leakage_rejected(self):
        self.rows[2700]['group_id'] = self.rows[0]['group_id']
        write(self.out / 'bad.jsonl', self.rows)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            load_training(self.out / 'bad.jsonl')

    def test_uncertainty_adds_review(self):
        teacher = read(self.out / 'teacher.jsonl')
        teacher[0].update(label=None, uncertain=True)
        write(self.out / 'teacher.jsonl', teacher)
        review(self.out, .2, 42)
        queue = {r['id']: r for r in read(self.out / 'review.jsonl')}
        self.assertIn('uncertain_or_unclassifiable', queue['0']['review_reasons'])


if __name__ == '__main__':
    unittest.main()

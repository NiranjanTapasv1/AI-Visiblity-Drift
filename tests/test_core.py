import os
import sys
import unittest

# ensure src is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)

import geo_drift_tracker as gdt


class TestParsingAndAggregation(unittest.TestCase):

    def test_parse_response_basic(self):
        text = (
            "1. AcmeCRM - Great for startups (positive)\n"
            "2. CloudSale - Affordable and simple (neutral)\n"
            "3. BizHawk - Powerful (negative)"
        )
        parsed = gdt.parse_response(text)
        # Expect three items with correct names and sentiments
        names = [p[0] for p in parsed]
        sentiments = [p[2] for p in parsed]
        self.assertIn('AcmeCRM', names)
        self.assertIn('CloudSale', names)
        self.assertIn('BizHawk', names)
        self.assertIn('positive', sentiments)

    def test_aggregate_runs_counts_and_stability(self):
        runs = []
        runs.append([('A', 1, 'positive'), ('B', 2, 'neutral')])
        runs.append([('A', 2, 'positive'), ('B', 1, 'neutral')])
        runs.append([('A', 1, 'positive')])
        df = gdt.aggregate_runs(runs, n_runs=3)
        # A should have freq 3, B freq 2
        row_a = df[df['brand'] == 'A'].iloc[0]
        row_b = df[df['brand'] == 'B'].iloc[0]
        self.assertEqual(row_a['mention_frequency'], 3)
        self.assertEqual(row_b['mention_frequency'], 2)
        # stability should be between 0 and 1
        self.assertGreaterEqual(row_a['stability'], 0)
        self.assertLessEqual(row_a['stability'], 1)


if __name__ == '__main__':
    unittest.main()

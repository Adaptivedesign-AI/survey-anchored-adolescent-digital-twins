"""Offline regression checks for the narrative lookup; no respondent data loaded."""
import csv
import itertools
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/yrbs'

class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (DATA / 'Final_Mapping_Table__Q1_Q107_.csv').open() as source:
            cls.records = list(csv.DictReader(source))
        cls.lookup = {(r['question'], r['code']): r['prompt_fragment'] for r in cls.records}
        cls.questions = json.loads((DATA / 'questions_107_converted.json').read_text())

    def test_unique_nonempty_keys_and_fragments(self):
        self.assertEqual(len(self.records), len(self.lookup))
        for row in self.records:
            self.assertTrue(row['prompt_fragment'].strip(), row)
            self.assertTrue(row['prompt_fragment'].endswith('.'), row)

    def test_covers_all_and_only_valid_categorical_options(self):
        expected=set()
        for q in self.questions:
            if q['type']=='numeric':
                continue  # Q6 and Q7 are handled by the profile generator.
            if q['id']=='Q5':
                for n in range(1,6):
                    expected.update(('Q5',''.join(c)) for c in itertools.combinations('ABCDE',n))
                continue
            options=re.findall(r'(?m)^[A-Z0-9]+\. (.+)$',q['text'])
            expected.update((q['id'],str(i)) for i in range(1,len(options)+1))
        self.assertEqual(expected,set(self.lookup))

    def test_correct_question_constructs(self):
        for q,required,forbidden in [('Q93','hallucinogenic','transportation'),('Q97','muscles','teeth'),('Q98','sunburn','mental health'),('Q99','basic needs','school')]:
            for (question,_),fragment in self.lookup.items():
                if question==q:
                    self.assertIn(required,fragment)
                    self.assertNotIn(forbidden,fragment)

    def test_time_windows_and_qualifiers(self):
        for c in ['1','2','3']:
            self.assertNotIn('12 months',self.lookup['Q81',c])
            self.assertIn('donating blood',self.lookup['Q81',c])
        self.assertIn('not during the past 3 months',self.lookup['Q59','2'])
        self.assertIn('before',self.lookup['Q60','2'])
        self.assertIn('excluding',self.lookup['Q13','1'])
        self.assertIn('almost every day',self.lookup['Q26','2'])
        self.assertIn('stopped doing',self.lookup['Q26','2'])
        self.assertIn('past 30 days',self.lookup['Q34','2'])
        self.assertIn('past 30 days',self.lookup['Q44','2'])
        self.assertIn('excluding diet',self.lookup['Q74','1'])
        self.assertIn('physical, mental, or emotional',self.lookup['Q106','2'])

    def test_missing_is_not_a_fact(self):
        self.assertNotIn(('Q87','8'),self.lookup)
        self.assertTrue(all(c.strip() and c!='Missing' for _,c in self.lookup))
        self.assertNotIn('as missing',' '.join(self.lookup.values()))

    def test_two_classroom_conversions(self):
        self.assertIn('60 minutes',self.lookup['Q76','6'])
        self.assertIn('on 5 days',self.lookup['Q76','6'])
        self.assertIn('past 7 days',self.lookup['Q76','6'])
        self.assertIn('past 12 months',self.lookup['Q78','2'])
        self.assertIn('1 sports team',self.lookup['Q78','2'])

if __name__=='__main__':
    unittest.main()

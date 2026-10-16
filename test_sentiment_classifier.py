import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class ClassifierTests(unittest.TestCase):
    def test_classifies_review_with_structured_openai_response(self):
        import sentiment_classifier as app
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=app.SentimentResult(sentiment='positive', reason='The reviewer liked the gift.'))
        review = {'title': 'Great gift', 'text': 'Always appreciated.', 'rating': 5.0}
        result = app.classify_review(client, review, model='gpt-4.1-mini')
        self.assertEqual(result.sentiment, 'positive')
        call = client.responses.parse.call_args.kwargs
        self.assertEqual(call['model'], 'gpt-4.1-mini')
        self.assertFalse(call['store'])
        self.assertIn('Always appreciated.', call['input'][0]['content'])
        self.assertIn('5.0', call['input'][0]['content'])
        self.assertIs(call['text_format'], app.SentimentResult)

    def test_cli_streams_limited_reviews_to_output(self):
        import json
        import tempfile
        from pathlib import Path
        import sentiment_classifier as app
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'reviews.jsonl'
            output = Path(directory) / 'labels.jsonl'
            source.write_text('\n'.join(json.dumps({'title': 'Bad', 'text': 'Did not work', 'rating': 1}) for _ in range(3)), encoding='utf-8')
            client = Mock()
            client.responses.parse.return_value = SimpleNamespace(output_parsed=app.SentimentResult(sentiment='negative', reason='Did not work'))
            self.assertEqual(app.main(['--input', str(source), '--output', str(output), '--limit', '2'], client=client), 0)
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['sentiment'], 'negative')
            self.assertEqual(rows[0]['source_line'], 1)
            self.assertEqual(client.responses.parse.call_count, 2)
            self.assertEqual(app.main(['--input', str(source), '--output', str(output)], client=client), 1)

    def test_refusal_is_not_silently_labeled(self):
        import sentiment_classifier as app
        client = Mock()
        client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        with self.assertRaises(ValueError):
            app.classify_review(client, {'title': '', 'text': '', 'rating': 3}, 'gpt-4.1-mini')

    def test_dry_run_validates_without_calls_or_output(self):
        import json
        import tempfile
        from pathlib import Path
        import sentiment_classifier as app
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.jsonl', Path(directory) / 'out.jsonl'
            source.write_text(json.dumps({'title': '', 'text': '', 'rating': 3}) + '\n', encoding='utf-8')
            client = Mock()
            self.assertEqual(app.main(['--input', str(source), '--output', str(output), '--dry-run'], client=client), 0)
            client.responses.parse.assert_not_called()
            self.assertFalse(output.exists())
            source.write_text(json.dumps({'title': '', 'text': '', 'rating': 9}) + '\n', encoding='utf-8')
            self.assertEqual(app.main(['--input', str(source), '--dry-run'], client=client), 1)


if __name__ == '__main__':
    unittest.main()

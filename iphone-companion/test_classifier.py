import json
import unittest
from unittest.mock import Mock, patch

from classifier import InboxClassifier


class ClassifierTests(unittest.TestCase):
    def classifier(self, uncertain):
        classifier = InboxClassifier()
        classifier.router = Mock(identity='local-model')
        classifier.router.route.return_value = ({'label': 'episodic', 'working_state': True,
            'uncertain': uncertain, 'confidence': .6 if uncertain else .95,
            'threshold': .82, 'source': 'classifier'}, None)
        return classifier

    @staticmethod
    def item(previous=None):
        return {'text': 'I finished the application. Nothing remains to do.', 'note': '',
                'url': '', 'classification': previous}

    @patch('classifier.complete')
    def test_low_confidence_review_retains_local_prediction(self, complete):
        complete.return_value = json.dumps({'items': [{'id': '0', 'label': 'episodic',
            'working_state': False, 'uncertain': False, 'reason': 'Completed event'}]})
        result = self.classifier(True)(self.item())
        self.assertEqual(result['source'], 'deepseek')
        self.assertFalse(result['working_state'])
        self.assertFalse(result['uncertain'])
        self.assertTrue(result['local_prediction']['working_state'])
        self.assertEqual(result['confidence'], .6)
        complete.assert_called_once()

    @patch('classifier.complete')
    def test_confident_prediction_does_not_call_cloud(self, complete):
        self.assertEqual(self.classifier(False)(self.item())['source'], 'classifier')
        complete.assert_not_called()

    @patch('classifier.complete')
    def test_review_keeps_all_applicable_labels(self, complete):
        annotation = {'id': '0', 'label': 'episodic', 'labels': ['episodic', 'semantic'],
                      'working_state': False, 'uncertain': False, 'reason': 'An event and a preference'}
        complete.return_value = json.dumps({'items': [annotation]})
        result = self.classifier(True)(self.item())
        self.assertEqual(result['labels'], ['episodic', 'semantic'])
        self.assertFalse(result['uncertain'])
        self.assertIn('同一条消息可以有多个标签', complete.call_args.args[0][0]['content'])
        annotation['labels'] = ['semantic']
        complete.return_value = json.dumps({'items': [annotation]})
        self.assertEqual(self.classifier(True)(self.item())['fallback_status'], 'failed')

    @patch('classifier.complete')
    def test_failure_or_invalid_answer_preserves_uncertainty(self, complete):
        for answer in ('not json', '{"items":[]}', '{"items":[{"id":"0","label":"invalid"}]}'):
            complete.return_value = answer
            result = self.classifier(True)(self.item())
            self.assertEqual(result['fallback_status'], 'failed')
            self.assertTrue(result['uncertain'])
            self.assertEqual(result['label'], 'episodic')
        complete.side_effect = TimeoutError()
        self.assertEqual(self.classifier(True)(self.item())['fallback_status'], 'failed')

    @patch('classifier.complete')
    def test_existing_low_confidence_record_can_retry_without_encoder(self, complete):
        complete.return_value = json.dumps({'items': [{'id': '0', 'label': None,
            'working_state': False, 'uncertain': True, 'reason': 'Insufficient context'}]})
        previous = {'status': 'classified', 'label': 'episodic', 'working_state': True,
                    'confidence': .6, 'source': 'classifier', 'uncertain': True}
        result = InboxClassifier()(self.item(previous))
        self.assertEqual(result['source'], 'deepseek')
        self.assertTrue(result['uncertain'])
        self.assertIsNone(result['label'])

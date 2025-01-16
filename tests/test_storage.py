import unittest
from unittest.mock import patch, mock_open

from storage.local_storage import LocalStorageProvider
from storage.s3_storage import S3StorageProvider


class TestLocalStorageProvider(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isdir', return_value=False)
    def test_write(self, mock_isdir, mock_open):
        provider = LocalStorageProvider()
        provider.write('test_path', 'test_data')
        mock_open.assert_called_once_with('test_path', 'w')
        mock_open().write.assert_called_once_with('test_data')

    @patch('builtins.open', new_callable=mock_open, read_data='test_data')
    @patch('os.path.isdir', return_value=False)
    def test_read(self, mock_isdir, mock_open):
        provider = LocalStorageProvider()
        result = list(provider.read('test_path'))
        mock_open.assert_called_once_with('test_path', 'r')
        self.assertEqual(result, ['test_data'])


class TestS3StorageProvider(unittest.TestCase):

    @patch('boto3.client')
    def setUp(self, mock_boto_client):
        self.mock_s3_client = mock_boto_client.return_value
        self.provider = S3StorageProvider('test_bucket')

    def test_write(self):
        self.provider.write('/tmp/test_path.data', 'test_data')
        self.mock_s3_client.put_object.assert_called_once_with(Bucket='test_bucket', Key='/tmp/test_path.data',
                                                               Body='test_data')


# class TestPDFReader(unittest.TestCase):
#
#     @patch('storage.storage.PdfReader')
#     def test_read_pdf(self, mock_pdf_reader):
#         mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: 'page_text')]
#         mock_storage_provider = MagicMock()
#         mock_storage_provider.read.return_value = [b'pdf_data']
#         reader = PDFReader(mock_storage_provider)
#         result = reader.read_pdf('test_path')
#         self.assertEqual(result, ['page_text'])

if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.core.exceptions import SourceErrors
from app.services.document_service import DocumentService, UploadValidationError


class FictionBookTests(unittest.TestCase):
    def extract(self, xml, encoding="utf-8"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.fb2"
            path.write_bytes(xml.encode(encoding))
            return DocumentService.extract_blocks(str(path), ".fb2")

    def test_reading_order_inline_markup_notes_tables_and_metadata(self):
        blocks = self.extract('''<?xml version="1.0" encoding="UTF-8"?>
        <FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
          <description><title-info><book-title>Физика</book-title>
            <author><first-name>Иван</first-name><last-name>Иванов</last-name></author>
          </title-info><document-info><id>NOT_READING_TEXT</id></document-info></description>
          <body><section><title><p>Глава 1</p></title>
            <p>Сила <emphasis>равна</emphasis> массе.</p>
            <section><p>Второй абзац.</p></section>
            <table><tr><td>Масса</td><td>кг</td></tr></table>
            <poem><stanza><v>Строка стиха</v></stanza></poem>
          </section></body>
          <body name="notes"><section><p>Примечание.</p></section></body>
          <binary id="cover">NOT_READING_TEXT</binary>
        </FictionBook>''')
        self.assertEqual([b.text for b in blocks], [
            "Физика", "Иван Иванов", "Глава 1", "Сила равна массе.",
            "Второй абзац.", "Масса | кг", "Строка стиха", "Примечание.",
        ])
        self.assertEqual([b.order for b in blocks], list(range(8)))
        self.assertTrue(all(b.source == "fb2" and b.page == 1 for b in blocks))

    def test_xml_declared_legacy_encoding_and_namespace_free_books(self):
        blocks = self.extract('''<?xml version="1.0" encoding="windows-1251"?>
        <FictionBook><body><section><p>Наука и образование</p></section></body></FictionBook>''', "cp1251")
        self.assertEqual(blocks[0].text, "Наука и образование")

    def test_rejects_malformed_wrong_root_empty_and_external_entities(self):
        cases = [
            "<FictionBook><body>",
            "<html><body><p>Not a book</p></body></html>",
            '<FictionBook xmlns="https://wrong.example"><body><p>Text</p></body></FictionBook>',
            "<FictionBook><description><title-info><book-title>Only a title</book-title></title-info></description><body/></FictionBook>",
            '<!DOCTYPE FictionBook [<!ENTITY secret SYSTEM "file:///etc/passwd">]><FictionBook><body><p>&secret;</p></body></FictionBook>',
            '<!DOCTYPE FictionBook [<!ENTITY a "expanded">]><FictionBook><body><p>&a;</p></body></FictionBook>',
        ]
        for xml in cases:
            with self.subTest(xml=xml), self.assertRaises(UploadValidationError) as error:
                self.extract(xml)
            self.assertEqual(error.exception.error_code, SourceErrors.INVALID_UPLOAD)

    def test_upload_accepts_fb2_xml_and_generic_mime(self):
        for mime in ("application/x-fictionbook+xml", "application/xml", "text/xml", "application/octet-stream", ""):
            with self.subTest(mime=mime):
                upload = SimpleNamespace(filename="Книга.FB2", content_type=mime, size=100)
                self.assertEqual(DocumentService.validate_upload_file(upload), ".fb2")
        with self.assertRaises(UploadValidationError):
            DocumentService.validate_upload_file(SimpleNamespace(filename="book.fb2", content_type="image/png", size=100))


if __name__ == "__main__":
    unittest.main()

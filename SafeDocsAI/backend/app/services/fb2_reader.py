"""Extract reading text from FictionBook XML without resolving entities."""

from lxml import etree

from app.services.hybrid_chunker import TextBlock


FB2_NAMESPACE = "http://www.gribuser.ru/xml/fictionbook/2.0"


def extract_fb2_blocks(file_path: str) -> list[TextBlock]:
    parser = etree.XMLParser(
        resolve_entities=False, load_dtd=False, no_network=True, recover=False,
    )
    try:
        tree = etree.parse(file_path, parser)
    except etree.XMLSyntaxError as exc:
        raise ValueError("Invalid FictionBook XML") from exc

    if tree.docinfo.doctype:
        raise ValueError("DTD declarations are not allowed in FB2 files")
    root = tree.getroot()
    root_name = etree.QName(root)
    if root_name.localname != "FictionBook" or root_name.namespace not in (None, FB2_NAMESPACE):
        raise ValueError("Expected a FictionBook document")

    prefix = f"{{{root_name.namespace}}}" if root_name.namespace else ""
    blocks: list[TextBlock] = []

    def add(text: str) -> None:
        text = " ".join(text.split())
        if text:
            # FB2 has sections but no stable printed page numbers.
            blocks.append(TextBlock(text=text, page=1, order=len(blocks), source="fb2"))

    info = root.find(f"{prefix}description/{prefix}title-info")
    if info is not None:
        title = info.find(f"{prefix}book-title")
        if title is not None:
            add("".join(title.itertext()))
        for author in info.findall(f"{prefix}author"):
            add(" ".join("".join(part.itertext()) for part in author
                         if part.tag in {prefix + name for name in
                                         ("first-name", "middle-name", "last-name", "nickname")}))

    bodies = root.findall(f"{prefix}body")
    if not bodies:
        raise ValueError("FictionBook document has no body")
    metadata_count = len(blocks)
    reading_blocks = {prefix + name for name in ("p", "v", "subtitle", "text-author", "date")}
    for body in bodies:
        for element in body.iter():
            if element.tag in reading_blocks:
                add("".join(element.itertext()))
            elif element.tag == prefix + "tr":
                add(" | ".join("".join(cell.itertext()).strip() for cell in element))

    # Only body text counts as a readable book; a title/cover alone is not enough.
    if len(blocks) == metadata_count:
        raise ValueError("FictionBook document contains no reading text")
    return blocks

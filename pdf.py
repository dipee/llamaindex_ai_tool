from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import os
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print(f"building index {index_name}")
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))
    return index

pdf_path  = os.path.join('data', 'Nepal.pdf')
nepal_pdf = PDFReader().load_data(file=pdf_path)
nepal_index = get_index(nepal_pdf, 'nepal')
nepal_engine = nepal_index.as_query_engine()

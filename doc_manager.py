
from pathlib import Path

class DocumentManager:
    def __init__(self, doc_dir="./docs"):
        self.doc_dir = Path(doc_dir)
        self.doc_dir.mkdir(exist_ok=True)

    def list_documents(self):
        return [f.name for f in self.doc_dir.iterdir() if f.is_file()]

    def delete_documents(self, filenames):
        deleted = []
        for fname in filenames:
            file_path = self.doc_dir / fname
            try:
                if file_path.exists():
                    file_path.unlink()
                    deleted.append(fname)
                    print(f"Deleted: {fname}")
            except Exception as e:
                print(f"Error deleting {fname}: {e}")
        return deleted

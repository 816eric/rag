import os
import shutil
import json
from pathlib import Path
import config.config as config


DOC_DIR = "./docs"
MANIFEST_FILE = "embedded_files.json"


class DocumentManager:
    def __init__(self, docs_path=DOC_DIR, chroma_path=config.DB_DIR):
        self.docs_path = Path(docs_path)
        self.chroma_path = Path(chroma_path)
        self.manifest_path = self.chroma_path / MANIFEST_FILE

        self.docs_path.mkdir(exist_ok=True)
        self.chroma_path.mkdir(exist_ok=True)

    def _load_manifest(self):
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                files = json.load(f)
                print(f"Loaded manifest with files: {files}.")
                return files
        return []

    def _save_manifest(self, filenames):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(filenames, f)

    def list_documents(self):
        """List all document filenames that have been embedded, based on the manifest."""
        files = self._load_manifest()
        print(f"Listing documents: {files}")
        return files

    def add_to_manifest(self, filenames):
        """Add filenames to the embedded file manifest."""
        manifest = self._load_manifest()
        updated = False
        for filename in filenames:
            #base_filename = os.path.basename(filename)
            if filename not in manifest:
                print(f"Adding {filename} to manifest.")
                manifest.append(filename)
                updated = True
        if updated:
            self._save_manifest(manifest)
        print(f"Updated manifest with {len(filenames)} new files: {filenames}")
    
    def add_documents(self, filenames):
        """Add new document files to the docs directory and update the manifest.

        Returns only the paths of the files just added, not the full manifest -
        callers use this to know what to embed, and re-embedding every previously
        added document on each call would duplicate their chunks in the vector store.
        """
        added_paths = []
        for file in filenames:
            filename = os.path.basename(file.name)
            save_path = os.path.join(self.docs_path, filename)
            shutil.copy(file.name, save_path)
            # Update manifest
            self.add_to_manifest([save_path])
            added_paths.append(save_path)
            print(f"Added document: {save_path}")

        return added_paths

    def add_external_documents(self, file_paths):
        """Register documents by reference (no copy) - used for folder-based embedding
        where files may be large in number or size and should stay in place."""
        self.add_to_manifest(file_paths)
        return self.list_documents()

    def delete_documents_manifest(self, filenames):
        """Delete selected document files and remove them from the manifest.
        Only files that live inside docs_path (uploaded copies) are removed from disk;
        externally referenced files (e.g. from folder embedding) are only unlinked from the manifest."""
        for name in filenames:
            file_path = self.docs_path / name  # Use Path object
            if os.path.isabs(name):
                file_path = Path(name)
            try:
                inside_docs_path = file_path.resolve().is_relative_to(self.docs_path.resolve())
            except (OSError, ValueError):
                inside_docs_path = False
            if inside_docs_path and file_path.exists():
                print(f"Deleted document: {file_path}")
                file_path.unlink()
            else:
                print(f"Removed reference to external document (file left on disk): {file_path}")

        # Update manifest
        manifest = self._load_manifest()
        manifest = [f for f in manifest if os.path.basename(f) not in [os.path.basename(name) for name in filenames]]
        print(f"Updated manifest after deletion: {manifest}")
        self._save_manifest(manifest)

        print(f"Deleted documents and updated manifest: {filenames}")

        return manifest

"""
Custom dataset for IFCB plankton images with taxonomic superclass mapping.
"""
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class IFCBTrainDataset(Dataset):
    """
    Dataset for IFCB images that:
    1. Loads only images in the training split (from ifcb_train.csv)
    2. Maps folder names (species) to class indices
    3. Maps classes to superclasses (Phyla) from taxonomy
    """
    
    def __init__(self, data_path, train_csv_path, records_csv_path, transform=None,
                 image_embeddings=None):
        """
        Args:
            data_path: Path to root directory containing species subdirectories
            train_csv_path: Path to ifcb_train.csv
            records_csv_path: Path to ifcb_records.csv (for taxonomy)
            transform: torchvision transforms to apply to images
            image_embeddings: optional path to a .npz holding per-image LoRA-CLIP embeddings
                (keys clip_emb_image + clip_emb_image_keys, from make_clip_embeddings.py
                --images-embed). When given, __getitem__ returns (img, class_idx, emb) so the
                model can condition on each image's OWN embedding rather than a class
                prototype. Without it the dataset is unchanged and returns (img, class_idx).
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load CSVs
        self.train_df = pd.read_csv(train_csv_path)
        self.records_df = pd.read_csv(records_csv_path)
        
        # Key the training split on (Folder, image), NOT on the filename alone.
        # IFCB filenames encode instrument + timestamp (D20240312T135943_IFCB191_00092.png)
        # and are NOT unique across classes: 23 names appear in both ifcb_train.csv and
        # ifcb_test.csv under different folders, and 35 are duplicated within train itself.
        # Matching on the name alone therefore pulled 23 genuine TEST images into training,
        # each labelled with whatever folder it physically sits in (e.g. Detritus/...00092.png
        # was included because a different image of that name is a training image under
        # Meuniera_membranacea_single). Keying on the pair yields exactly the 59344 rows in
        # ifcb_train.csv.
        self.train_keys = set(zip(self.train_df['Folder'], self.train_df['image']))
        
        # Create superclass mapping from Phylum
        unique_phyla = sorted(self.records_df['Phylum'].dropna().unique())
        self.phylum_to_superclass_id = {phylum: i for i, phylum in enumerate(unique_phyla)}
        
        # Create mapping from folder name to superclass ID
        self.folder_to_superclass = {}
        for folder in self.records_df['Folder'].unique():
            phylum = self.records_df[self.records_df['Folder'] == folder]['Phylum'].iloc[0]
            if pd.notna(phylum):
                self.folder_to_superclass[folder] = self.phylum_to_superclass_id[phylum]
        
        # Build list of (image_path, class_idx, folder_name)
        self.samples = []
        self.class_to_idx = {}
        self.class_to_folder = {}
        class_idx = 0
        
        for folder in sorted(self.data_path.iterdir()):
            if folder.is_dir():
                self.class_to_idx[folder.name] = class_idx
                self.class_to_folder[class_idx] = folder.name
                
                for img_path in folder.glob('*.png'):
                    if (folder.name, img_path.name) in self.train_keys:
                        self.samples.append((str(img_path), class_idx, folder.name))
                
                class_idx += 1
        
        # Store metadata
        self.num_classes = class_idx
        self.num_superclasses = len(self.phylum_to_superclass_id)
        self.class_names = [self.class_to_folder[i] for i in range(self.num_classes)]
        self.superclass_names = [sc for sc, _ in sorted(self.phylum_to_superclass_id.items(), 
                                                         key=lambda x: x[1])]

        # Optional per-image CLIP embeddings, looked up by "<Folder>/<file>.png".
        self.image_embeddings = None
        if image_embeddings is not None:
            import numpy as np
            z = np.load(image_embeddings, allow_pickle=True)
            if "clip_emb_image" not in z:
                raise ValueError(
                    f"{image_embeddings} has no clip_emb_image — regenerate it with "
                    f"make_clip_embeddings.py --images-embed")
            emb = z["clip_emb_image"]
            keys = {k: i for i, k in enumerate(map(str, z["clip_emb_image_keys"]))}
            # Resolve every sample up-front so a missing embedding is a startup error rather
            # than a KeyError thousands of steps into training.
            rows = []
            for img_path, _, folder_name in self.samples:
                k = f"{folder_name}/{Path(img_path).name}"
                if k not in keys:
                    raise ValueError(f"no embedding for {k} in {image_embeddings}")
                rows.append(keys[k])
            self.image_embeddings = torch.from_numpy(emb[rows].astype("float32"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx, folder_name = self.samples[idx]
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        if self.image_embeddings is not None:
            return img, class_idx, self.image_embeddings[idx]
        return img, class_idx
    
    def get_superclass(self, class_idx):
        """Get superclass ID for a given class index."""
        folder_name = self.class_to_folder.get(class_idx)
        if folder_name:
            return self.folder_to_superclass.get(folder_name, 0)
        return 0
    
    def get_class_name(self, class_idx):
        """Get folder/species name for a class index."""
        return self.class_to_folder.get(class_idx, "Unknown")
    
    def get_superclass_name(self, superclass_idx):
        """Get Phylum name for a superclass index."""
        if 0 <= superclass_idx < len(self.superclass_names):
            return self.superclass_names[superclass_idx]
        return "Unknown"
    
    def get_stats(self):
        """Return dataset statistics."""
        stats = {
            'num_images': len(self.samples),
            'num_classes': self.num_classes,
            'num_superclasses': self.num_superclasses,
            'class_names': self.class_names,
            'superclass_names': self.superclass_names,
        }
        
        # Count images per superclass
        superclass_counts = {}
        for _, class_idx, folder_name in self.samples:
            sc_idx = self.folder_to_superclass.get(folder_name, 0)
            superclass_counts[sc_idx] = superclass_counts.get(sc_idx, 0) + 1
        stats['superclass_counts'] = superclass_counts
        
        return stats
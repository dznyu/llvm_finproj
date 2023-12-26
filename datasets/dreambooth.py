import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data


class DreamBoothDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.jpg'):
                    self.paths.append((os.path.join(cls_dir, filename), cls))

        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, class_text = self.paths[index]
        images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        texts = data.load_and_transform_text([class_text], self.device)

        return images, ModalityType.VISION, texts, ModalityType.TEXT



class ESCaudio_FT(Dataset):
    ### image-audio data
    def __init__(self, root_dir: str = "/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master",
                 transform: Optional[Callable] = None,
                 split: str = 'train', 
                 flavor: str = 'esc10',
                 # train_size: float = 0.8, 
                 test_fold: int = 5,
                 random_seed: int = 42, 
                 device: str = 'cpu'):

        assert flavor in ("esc10", "esc50", "all")

        self.flavor = flavor
        
        self.root_dir = root_dir
        self.transform = transform
        self.test_fold = test_fold
        self.device = device

        self.paths = None
        self.all_paths = []
        self.esc_map = pd.read_csv(os.path.join(self.root_dir, "meta", "esc50.csv"))
        self.wavdir = os.path.join(self.root_dir, "audio")

        self.train_paths = []
        self.test_paths = []
        
        if self.flavor == "esc10":
            self.esc10_map = self.esc_map[self.esc_map["esc10"] == True]
            self.classes = list(self.esc10_map["category"].unique())
            self.class_to_idx = dict(self.esc10_map[['category','target']].drop_duplicates().reset_index(drop=True).values)
            self.melspecdir = os.path.join(self.root_dir, "melspec_all")
            
            for idx, row in self.esc10_map.iterrows():
                audwav_path = os.path.join(self.wavdir, row["filename"])
                melimg_path = os.path.join(self.melspecdir, row["filename"] + ".jpg")
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))
            
        elif self.flavor == "esc50":
            
            self.esc50_map = self.esc_map[self.esc_map["esc10"] == False]            
            self.classes = list(self.esc50_map["category"].unique())
            self.class_to_idx = dict(self.esc50_map[['category','target']].drop_duplicates().reset_index(drop=True).values)
            self.melspecdir = os.path.join(self.root_dir, "melspec_all/esc50_specimgs")

            for idx, row in self.esc50_map.iterrows():
                audwav_path = os.path.join(self.wavdir, row["filename"])
                melimg_path = os.path.join(self.melspecdir, row["filename"] + ".jpg")
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))

        
        elif self.flavor == "all":
            self.classes = list(self.esc_map["category"].unique())
            self.class_to_idx = dict(self.esc_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            for idx, row in self.esc_map.iterrows():
                if row["esc10"] == True:
                    melimg_path = os.path.join(os.path.join(self.root_dir, "melspec_all/esc50_specimgs"), 
                                               row["filename"] + ".jpg")
                else:
                    melimg_path = os.path.join(os.path.join(self.root_dir, "melspec_all/esc50_specimgs"), 
                                               row["filename"] + ".jpg")
                audwav_path = os.path.join(self.wavdir, row["filename"])
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))
        
        if split == 'train':
            self.paths = self.train_paths
        elif split == 'test':
            self.paths = self.test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        audwav_path, melspecimg_path, class_text = self.train_paths[index]
        images = data.load_and_transform_vision_data([melspecimg_path], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        texts = data.load_and_transform_text([class_text], self.device)
        audio = data.load_and_transform_audio_data([audwav_path], self.device)

        return audio, ModalityType.AUDIO, images, ModalityType.VISION, texts, ModalityType.TEXT


class DreamBoothDataset_decode(Dataset):
    ### image-text data
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and (d != ".ipynb_checkpoints")]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_dir):
                if filename.endswith('.jpg'):
                    self.paths.append((os.path.join(cls_dir, filename), cls))

        # Split dataset
        train_paths, test_paths = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_paths
        elif split == 'test':
            self.paths = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, class_text = self.paths[index]
        images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=False)

        if self.transform is not None:
            image = images[0]
            images = self.transform(image)

        # texts = data.load_and_transform_text([class_text], self.device)

        return images, ModalityType.VISION


class ESCaudio_decode(Dataset):
    ### image-audio data
    def __init__(self, root_dir: str = "/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master",
                 transform: Optional[Callable] = None,
                 split: str = 'train', 
                 flavor: str = 'esc10',
                 # train_size: float = 0.8, 
                 test_fold: int = 5,
                 random_seed: int = 42, 
                 device: str = 'cpu'):

        assert flavor in ("esc10", "esc50", "all")

        self.flavor = flavor
        
        self.root_dir = root_dir
        self.transform = transform
        self.test_fold = test_fold
        self.device = device

        self.paths = None
        self.all_paths = []
        self.esc_map = pd.read_csv(os.path.join(self.root_dir, "meta", "esc50.csv"))
        self.wavdir = os.path.join(self.root_dir, "audio")

        self.train_paths = []
        self.test_paths = []
        
        if self.flavor == "esc10":
            self.esc10_map = self.esc_map[self.esc_map["esc10"] == True]
            self.classes = list(self.esc10_map["category"].unique())
            self.class_to_idx = dict(self.esc10_map[['category','target']].drop_duplicates().reset_index(drop=True).values)
            self.melspecdir = os.path.join(self.root_dir, "melspec_all")
            
            for idx, row in self.esc10_map.iterrows():
                audwav_path = os.path.join(self.wavdir, row["filename"])
                melimg_path = os.path.join(self.melspecdir, row["filename"] + ".jpg")
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))
            
        elif self.flavor == "esc50":
            
            self.esc50_map = self.esc_map[self.esc_map["esc10"] == False]            
            self.classes = list(self.esc50_map["category"].unique())
            self.class_to_idx = dict(self.esc50_map[['category','target']].drop_duplicates().reset_index(drop=True).values)
            self.melspecdir = os.path.join(self.root_dir, "melspec_all/esc50_specimgs")

            for idx, row in self.esc50_map.iterrows():
                audwav_path = os.path.join(self.wavdir, row["filename"])
                melimg_path = os.path.join(self.melspecdir, row["filename"] + ".jpg")
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))

        
        elif self.flavor == "all":
            self.classes = list(self.esc_map["category"].unique())
            self.class_to_idx = dict(self.esc_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            for idx, row in self.esc_map.iterrows():
                if row["esc10"] == True:
                    melimg_path = os.path.join(os.path.join(self.root_dir, "melspec_all/esc50_specimgs"), 
                                               row["filename"] + ".jpg")
                else:
                    melimg_path = os.path.join(os.path.join(self.root_dir, "melspec_all/esc50_specimgs"), 
                                               row["filename"] + ".jpg")
                audwav_path = os.path.join(self.wavdir, row["filename"])
                class_name = row["category"]
                fold = row["fold"]

                self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audwav_path, melimg_path, class_name))
                else:
                    self.train_paths.append((audwav_path, melimg_path, class_name))
        
        if split == 'train':
            self.paths = self.train_paths
        elif split == 'test':
            self.paths = self.test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        audwav_path, _, _ = self.train_paths[index]
        
        # images = data.load_and_transform_vision_data([melspecimg_path], self.device, to_tensor=False)

        # if self.transform is not None:
        #     image = images[0]
        #     images = self.transform(image)

        # texts = data.load_and_transform_text([class_text], self.device)
        audio = data.load_and_transform_audio_data([audwav_path], self.device)

        return audio, ModalityType.AUDIO



class ESCaudio_multiEmbed(Dataset):
    ### image-audio data
    def __init__(self,
                 transform: Optional[Callable] = None,
                 split: str = 'train',
                 flavor: str = 'esc10',
                 # train_size: float = 0.8, 
                 test_fold: int = 5,
                 random_seed: int = 42, 
                 device: str = 'cpu'):

        assert flavor in ("esc10", "esc50", "all")

        self.flavor = flavor
        
        self.root_dir = "/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/"
        self.test_fold = test_fold
        self.device = device

        self.paths = None
        self.all_paths = {}
        self.esc_map = pd.read_csv(os.path.join(self.root_dir, "meta", "esc50.csv"))
        self.txtembed_dict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc_txtdict.pt")
        
        self.train_paths = []
        self.test_paths = []
        
        if self.flavor == "esc10":

            esc10_wavdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc10_wavdict.pt")
            esc10_specdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc10_specdict_compatwav.pt")

            self.esc10_map = self.esc_map[self.esc_map["esc10"] == True]
            self.classes = list(self.esc10_map["category"].unique())
            self.class_to_idx = dict(self.esc10_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            
            for idx, row in self.esc10_map.iterrows():
                id = row["filename"]
                class_name = row["category"]
                fold = row["fold"]

                audio_embed = esc10_wavdict[id]
                img_embed = esc10_specdict[id]

                txt_embed = self.txtembed_dict[class_name]
                
                self.all_paths[id] = (class_name, fold)
                # self.all_paths.append((audio_embed, img_embed))

                if fold == self.test_fold:
                    self.test_paths.append((audio_embed, img_embed, txt_embed))
                else:
                    self.train_paths.append((audio_embed, img_embed, txt_embed))
            
        elif self.flavor == "esc50":

            ## deal with only files that exist
            exist_esc50_f = [i for i in os.listdir("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/melspec_all/esc50_specimgs") if i.endswith(".jpg")]
            
            esc50_wavdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc50_wavdict.pt")
            esc50_specdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc50_specdict_compatwav.pt")
            
            self.esc50_map = self.esc_map[self.esc_map["esc10"] == False]            
            self.classes = list(self.esc50_map["category"].unique())
            self.class_to_idx = dict(self.esc50_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            for idx, row in self.esc50_map.iterrows():
                id = row["filename"]
                id_jpg = id + ".jpg"
                class_name = row["category"]
                fold = row["fold"]

                if id_jpg in exist_esc50_f:
                    audio_embed = esc50_wavdict[id]
                    img_embed = esc50_specdict[id]
                    txt_embed = self.txtembed_dict[class_name]

                self.all_paths[id] = (class_name, fold)                
                # self.all_paths.append((audwav_path, melimg_path, class_name))

                if fold == self.test_fold:
                    self.test_paths.append((audio_embed, img_embed, txt_embed))
                else:
                    self.train_paths.append((audio_embed, img_embed, txt_embed))

        
        elif self.flavor == "all":
            self.classes = list(self.esc_map["category"].unique())
            self.class_to_idx = dict(self.esc_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            esc10_wavdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc10_wavdict.pt")
            esc10_specdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc10_specdict_compatwav.pt")
            esc50_wavdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc50_wavdict.pt")
            esc50_specdict = torch.load("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/esc50_specdict_compatwav.pt")

            escall_wavdict = esc10_wavdict.copy()
            escall_specdict = esc10_specdict.copy()
            
            escall_wavdict.update(esc50_wavdict)
            escall_specdict.update(esc50_specdict)

            exist_esc50_f = [i for i in os.listdir("/scratch/dz1158/imagebind/datas/audio_esc50/ESC-50-master/melspec_all/esc50_specimgs") if i.endswith(".jpg")]
            
            self.classes = list(self.esc_map["category"].unique())
            self.class_to_idx = dict(self.esc_map[['category','target']].drop_duplicates().reset_index(drop=True).values)

            for idx, row in self.esc_map.iterrows():
                id = row["filename"]
                id_jpg = id + ".jpg"
                class_name = row["category"]
                fold = row["fold"]

                if row["esc10"] == True:
                    audio_embed = escall_wavdict[id]
                    img_embed = escall_specdict[id]
                    txt_embed = self.txtembed_dict[class_name]
                else:
                    if id_jpg in exist_esc50_f:
                        audio_embed = escall_wavdict[id]
                        img_embed = escall_specdict[id]
                        txt_embed = self.txtembed_dict[class_name]

                self.all_paths[id] = (class_name, fold)                

                if fold == self.test_fold:
                    self.test_paths.append((audio_embed, img_embed, txt_embed))
                else:
                    self.train_paths.append((audio_embed, img_embed, txt_embed))
            
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        audio_embed, img_embed, txt_embed = self.train_paths[index]
        # output = torch.concat([audio_embed, img_embed, class_text])
        # output = torch.concat([audio_embed, img_embed])
        
        return audio_embed, img_embed, txt_embed
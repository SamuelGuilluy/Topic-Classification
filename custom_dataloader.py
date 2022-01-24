from torch.utils.data import Dataset   
import os

class MyCustomDataset(Dataset):     
    def __init__(self, path_to_folder):
        self.path_to_folder = path_to_folder

    def __len__(self):   
        files = os.listdir(self.path_to_folder)
        num_text_files = 0
        for file in files:
            if(file[-3:] == "txt"):
                num_text_files +=1
        return num_text_files                                                            

    def __getitem__(self, i):  
        """ Get a item from a custom Pytorch Dataset given its index."""
        ## The format is 5 digit integer
        _tpl = os.path.join(self.path_to_folder,  "{:05d}")   
        ## open the file in a with segment in order to close it automatically
        with open( _tpl.format(i-1)+".txt", "r", encoding = 'utf-8') as f:
            ## get the sentence  
            sentence = f.readline()
            return sentence
        
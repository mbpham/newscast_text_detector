import numpy as np
import pandas as pd
from text_processor import TextProcessor

labels = np.load('labels_.npy', allow_pickle=True)
labels = pd.DataFrame(labels)
labels = labels.rename(columns={0: "Frame", 1: "Story title", 2: "Story subject", 3: "Om lidt", 4: "Direkte", 5: "Kort nyt", 6: "Location"})

tp = TextProcessor(labels)

cleaned_labels = tp.clean_labels()

cleaned_labels.to_csv("cleaned_labels.csv", index=False)
